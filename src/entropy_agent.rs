/// EntropyAgent — Adaptive Gossip Protocol (Rust Core)
///
/// Port from calibrated Python reference (stress_v13_eigenzeit.py).
/// Constants from v14.1 Dual-Sweep Calibration (218,700 simulations).
///
/// This module is compiled to a shared library (.so/.pyd) and distributed
/// without source code. The algorithm is proprietary IP.

// ─────────────────────────────────────────────────────────────────────────────
// Constants — v14.1 Calibration (Latency-Aware Sweep 2026-05-10)
// ─────────────────────────────────────────────────────────────────────────────
const ALPHA_UP: f64 = 0.50;
const ALPHA_DOWN: f64 = 0.04;
const K_MIN: u32 = 1;
const K_MAX: u32 = 8;
const ENTROPY_LOW_THRESHOLD: f64 = 0.08;
const ENTROPY_HIGH_THRESHOLD: f64 = 0.50;
const SATURATION_HALF_K_ROUNDS: u64 = 3;
const SATURATION_SLEEP_ROUNDS: u64 = 10;
const SATURATION_SLEEP_PROB: f64 = 0.30;
const LOW_ENTROPY_SLEEP_PROB: f64 = 0.05;
const THERMAL_SHOCK_DELTA_THRESHOLD: i64 = 15;
const THERMAL_OFFLINE_THRESHOLD_MS: f64 = 5000.0;
const THERMAL_MAX_COOLDOWN: u32 = 10;
const MAX_CONSECUTIVE_SLEEPS: u32 = 10;

// P3: Partition Detection
const VARIANCE_ALPHA: f64 = 0.1;
const PARTITION_VARIANCE_THRESHOLD: f64 = 0.001;
const PARTITION_STALE_ROUNDS: u64 = 20;

/// Simple LCG-based PRNG for deterministic, reproducible behavior.
/// Avoids dependency on external RNG crates.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    /// Returns a float in [0.0, 1.0).
    pub fn next_f64(&mut self) -> f64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }
}

/// EntropyAgent — adaptive gossip scheduler.
///
/// Determines how many gossip copies (k) to send each round based on
/// observed packet loss, state staleness, and thermal shock detection.
pub struct EntropyAgent {
    pub node_id: u64,

    // EWMA entropy estimate [0, 1]
    entropy_estimate: f64,
    current_k: u32,

    // Saturation tracking
    rounds_since_change: u64,
    last_state_total: i64,

    // Thermal shock (eigenzeit context)
    thermal_cooldown: u32,
    shock_events: u32,
    last_send_time_ms: f64,
    last_recv_monotonic_ms: f64,

    // Metrics
    sleep_count: u64,
    total_rounds: u64,
    send_count: u64,
    loss_count: u64,
    consecutive_sleeps: u32,

    // P3: Entropy variance (partition detection)
    entropy_variance: f64,
    entropy_mean: f64,

    // RNG
    rng: Rng,
}

impl EntropyAgent {
    pub fn new(node_id: u64, seed: u64) -> Self {
        Self {
            node_id,
            entropy_estimate: 0.0,
            current_k: K_MIN,
            rounds_since_change: 0,
            last_state_total: 0,
            thermal_cooldown: 0,
            shock_events: 0,
            last_send_time_ms: 0.0,
            last_recv_monotonic_ms: 0.0,
            sleep_count: 0,
            total_rounds: 0,
            send_count: 0,
            loss_count: 0,
            consecutive_sleeps: 0,
            entropy_variance: 0.0,
            entropy_mean: 0.0,
            rng: Rng::new(seed),
        }
    }

    /// Called when a gossip packet is received. Detects thermal shock
    /// via eigenzeit (proper time) differential.
    pub fn on_receive(&mut self, delta: i64, current_time_ms: f64) {
        if delta <= 0 {
            // No state change — update monotonic to prevent false thermal shock
            if current_time_ms.is_finite() {
                self.last_recv_monotonic_ms = self.last_recv_monotonic_ms.max(current_time_ms);
            }
            return;
        }

        self.rounds_since_change = 0;

        // Thermal Shock v13: Eigenzeit context
        if delta > THERMAL_SHOCK_DELTA_THRESHOLD {
            let time_since_last_send = current_time_ms - self.last_recv_monotonic_ms;
            if time_since_last_send > THERMAL_OFFLINE_THRESHOLD_MS {
                let cooldown = (delta / 3).clamp(0, THERMAL_MAX_COOLDOWN as i64) as u32;
                if cooldown > self.thermal_cooldown {
                    self.thermal_cooldown = cooldown;
                    self.shock_events += 1;
                }
            }
        }

        // Update monotonic AFTER thermal shock check
        if current_time_ms.is_finite() {
            self.last_recv_monotonic_ms = self.last_recv_monotonic_ms.max(current_time_ms);
        }
    }

    /// Called after each send attempt. Updates EWMA entropy estimate.
    pub fn on_packet_sent(&mut self, delivered: bool) {
        self.send_count += 1;
        if !delivered {
            self.loss_count += 1;
        }

        // Asymmetric EWMA (AIMD principle)
        if !delivered {
            self.entropy_estimate = ALPHA_UP * 1.0 + (1.0 - ALPHA_UP) * self.entropy_estimate;
        } else {
            self.entropy_estimate = (1.0 - ALPHA_DOWN) * self.entropy_estimate;
        }

        // NaN guard + clamp
        if !self.entropy_estimate.is_finite() {
            self.entropy_estimate = 0.0;
        } else {
            self.entropy_estimate = self.entropy_estimate.clamp(0.0, 1.0);
        }

        // P3: Entropy variance tracking (EWMA of mean-centered variance)
        self.entropy_mean =
            VARIANCE_ALPHA * self.entropy_estimate + (1.0 - VARIANCE_ALPHA) * self.entropy_mean;
        let deviation = self.entropy_estimate - self.entropy_mean;
        self.entropy_variance =
            VARIANCE_ALPHA * (deviation * deviation) + (1.0 - VARIANCE_ALPHA) * self.entropy_variance;
        if !self.entropy_variance.is_finite() {
            self.entropy_variance = 0.0;
        }
        if !self.entropy_mean.is_finite() {
            self.entropy_mean = 0.0;
        }

        // Adapt k based on entropy
        if self.entropy_estimate > ENTROPY_HIGH_THRESHOLD {
            self.current_k = (self.current_k + 1).min(K_MAX);
        } else if self.entropy_estimate < ENTROPY_LOW_THRESHOLD {
            self.current_k = self.current_k.saturating_sub(1).max(K_MIN);
        }
    }

    /// Returns the number of gossip copies for this round.
    /// 0 = skip this round (saturation brake or thermal shock).
    pub fn get_k(&mut self, current_time_ms: f64, state_total: i64) -> u32 {
        self.total_rounds += 1;

        // Update eigenzeit
        if current_time_ms.is_finite() && current_time_ms > self.last_send_time_ms {
            self.last_send_time_ms = current_time_ms;
            self.last_recv_monotonic_ms = self.last_recv_monotonic_ms.max(current_time_ms);
        }

        // Thermal shock cooldown (highest priority)
        if self.thermal_cooldown > 0 {
            self.thermal_cooldown -= 1;
            self.sleep_count += 1;
            self.consecutive_sleeps += 1;
            if self.consecutive_sleeps >= MAX_CONSECUTIVE_SLEEPS {
                self.consecutive_sleeps = 0;
                self.thermal_cooldown = 0;
                return K_MIN;
            }
            return 0;
        }

        // Saturation tracking
        if state_total > self.last_state_total {
            self.rounds_since_change = 0;
            self.last_state_total = state_total;
        } else {
            let cap = SATURATION_SLEEP_ROUNDS.max(PARTITION_STALE_ROUNDS) + 1;
            self.rounds_since_change = (self.rounds_since_change + 1).min(cap);
        }

        // Deep sleep on prolonged stagnation
        if self.rounds_since_change > SATURATION_SLEEP_ROUNDS {
            if self.rng.next_f64() < SATURATION_SLEEP_PROB {
                self.sleep_count += 1;
                self.consecutive_sleeps += 1;
                if self.consecutive_sleeps >= MAX_CONSECUTIVE_SLEEPS {
                    self.consecutive_sleeps = 0;
                    return K_MIN;
                }
                return 0;
            }
        }

        // Halve k on medium stagnation
        let mut effective_k = self.current_k;
        if self.rounds_since_change > SATURATION_HALF_K_ROUNDS {
            effective_k = (self.current_k / 2).max(K_MIN);
        }
        effective_k = effective_k.clamp(K_MIN, K_MAX);

        // Light sleep on low entropy
        if self.entropy_estimate < ENTROPY_LOW_THRESHOLD
            && self.rng.next_f64() < LOW_ENTROPY_SLEEP_PROB
        {
            self.sleep_count += 1;
            self.consecutive_sleeps += 1;
            if self.consecutive_sleeps >= MAX_CONSECUTIVE_SLEEPS {
                self.consecutive_sleeps = 0;
                return K_MIN;
            }
            return 0;
        }

        // Actual send — reset consecutive sleeps
        self.consecutive_sleeps = 0;
        effective_k
    }

    // ─── Introspection ───────────────────────────────────────────────────

    pub fn sleep_ratio(&self) -> f64 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.sleep_count as f64 / self.total_rounds as f64
        }
    }

    pub fn loss_rate(&self) -> f64 {
        if self.send_count == 0 {
            0.0
        } else {
            self.loss_count as f64 / self.send_count as f64
        }
    }

    pub fn entropy_estimate(&self) -> f64 {
        self.entropy_estimate
    }

    pub fn current_k_value(&self) -> u32 {
        self.current_k
    }

    pub fn shock_events(&self) -> u32 {
        self.shock_events
    }

    pub fn entropy_variance(&self) -> f64 {
        self.entropy_variance
    }

    /// Convergence estimation: T = -ln(0.01) / α_down × interval × entropy
    pub fn estimate_convergence_ms(&self, gossip_interval_ms: f64) -> f64 {
        if self.entropy_estimate < ENTROPY_LOW_THRESHOLD {
            return 0.0;
        }
        (4.605 / ALPHA_DOWN * gossip_interval_ms * self.entropy_estimate).ceil()
    }

    /// Partition detection heuristic.
    pub fn is_likely_partitioned(&self) -> bool {
        if self.thermal_cooldown > 0 || self.consecutive_sleeps > 0 {
            return false;
        }
        self.entropy_variance < PARTITION_VARIANCE_THRESHOLD
            && self.rounds_since_change > PARTITION_STALE_ROUNDS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_increases_on_loss() {
        let mut agent = EntropyAgent::new(0, 42);
        assert_eq!(agent.entropy_estimate(), 0.0);
        agent.on_packet_sent(false);
        assert!(agent.entropy_estimate() > 0.4);
    }

    #[test]
    fn test_entropy_decreases_on_delivery() {
        let mut agent = EntropyAgent::new(0, 42);
        // Drive entropy up
        for _ in 0..10 {
            agent.on_packet_sent(false);
        }
        let high = agent.entropy_estimate();
        // Deliver packets
        for _ in 0..100 {
            agent.on_packet_sent(true);
        }
        assert!(agent.entropy_estimate() < high);
    }

    #[test]
    fn test_k_adapts_upward() {
        let mut agent = EntropyAgent::new(0, 42);
        for _ in 0..20 {
            agent.on_packet_sent(false);
        }
        assert!(agent.current_k_value() > K_MIN);
    }

    #[test]
    fn test_thermal_shock() {
        let mut agent = EntropyAgent::new(0, 42);
        // Simulate long offline period
        agent.last_recv_monotonic_ms = 0.0;
        agent.on_receive(50, 10000.0); // delta=50, time gap=10s
        assert!(agent.shock_events() > 0);
        assert!(agent.thermal_cooldown > 0);
    }

    #[test]
    fn test_partition_detection() {
        let mut agent = EntropyAgent::new(0, 42);
        // Simulate stale state with zero variance
        agent.rounds_since_change = PARTITION_STALE_ROUNDS + 1;
        agent.entropy_variance = 0.0;
        agent.consecutive_sleeps = 0;
        agent.thermal_cooldown = 0;
        assert!(agent.is_likely_partitioned());
    }

    #[test]
    fn test_convergence_estimate_zero_when_converged() {
        let agent = EntropyAgent::new(0, 42);
        assert_eq!(agent.estimate_convergence_ms(500.0), 0.0);
    }

    #[test]
    fn test_floor_guard_prevents_infinite_sleep() {
        let mut agent = EntropyAgent::new(0, 42);
        agent.thermal_cooldown = 20; // Long cooldown
        let mut woke = false;
        for round in 0..15 {
            let k = agent.get_k(round as f64 * 500.0, 0);
            if k > 0 {
                woke = true;
                break;
            }
        }
        assert!(woke, "S6 floor guard must force wake-up within MAX_CONSECUTIVE_SLEEPS");
    }
}
