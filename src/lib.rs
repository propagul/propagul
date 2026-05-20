//! entropy_state_core — Compiled Python extension (PyO3)
//!
//! Exposes EntropyAgent and OrMap to Python as opaque, compiled objects.
//! Users can call methods but cannot inspect the implementation.

mod entropy_agent;
mod or_map;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "EntropyAgent")]
struct PyEntropyAgent {
    inner: entropy_agent::EntropyAgent,
}

#[pymethods]
impl PyEntropyAgent {
    #[new]
    #[pyo3(signature = (node_id, seed = 0))]
    fn new(node_id: u64, seed: u64) -> Self {
        Self {
            inner: entropy_agent::EntropyAgent::new(node_id, seed),
        }
    }


    #[pyo3(signature = (delta, current_time_ms))]
    fn on_receive(&mut self, delta: i64, current_time_ms: f64) {
        self.inner.on_receive(delta, current_time_ms);
    }

    fn on_packet_sent(&mut self, delivered: bool) {
        self.inner.on_packet_sent(delivered);
    }


    #[pyo3(name = "get_k", signature = (current_time_ms, state_total))]
    fn py_get_k(&mut self, current_time_ms: f64, state_total: i64) -> u32 {
        self.inner.get_k(current_time_ms, state_total)
    }


    #[getter]
    fn entropy(&self) -> f64 {
        self.inner.entropy_estimate()
    }


    #[getter]
    fn current_k(&self) -> u32 {
        self.inner.current_k_value()
    }


    #[getter]
    fn sleep_ratio(&self) -> f64 {
        self.inner.sleep_ratio()
    }


    #[getter]
    fn loss_rate(&self) -> f64 {
        self.inner.loss_rate()
    }


    #[getter]
    fn shock_events(&self) -> u32 {
        self.inner.shock_events()
    }


    #[pyo3(signature = (gossip_interval_ms = 500.0))]
    fn estimate_convergence_ms(&self, gossip_interval_ms: f64) -> f64 {
        self.inner.estimate_convergence_ms(gossip_interval_ms)
    }


    #[getter]
    fn is_partitioned(&self) -> bool {
        self.inner.is_likely_partitioned()
    }


    #[getter]
    fn entropy_variance(&self) -> f64 {
        self.inner.entropy_variance()
    }


    #[getter]
    fn node_id(&self) -> u64 {
        self.inner.node_id
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyAgent(node_id={}, entropy={:.4}, k={}, sleep_ratio={:.2}%)",
            self.inner.node_id,
            self.inner.entropy_estimate(),
            self.inner.current_k_value(),
            self.inner.sleep_ratio() * 100.0,
        )
    }
}


#[pyclass(name = "StateMap")]
struct PyStateMap {
    inner: or_map::OrMap,
}

#[pymethods]
impl PyStateMap {
    #[new]
    fn new(node_id: u64) -> Self {
        Self {
            inner: or_map::OrMap::new(node_id),
        }
    }


    fn set(&mut self, key: &str, value: &str) {
        self.inner.set(key, value);
    }


    fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).map(|s| s.to_string())
    }


    fn get_conflicts(&self, key: &str) -> Vec<String> {
        self.inner.get_all(key).iter().map(|s| s.to_string()).collect()
    }


    fn delete(&mut self, key: &str) {
        self.inner.delete(key);
    }


    fn keys(&self) -> Vec<String> {
        self.inner.keys().iter().map(|s| s.to_string()).collect()
    }


    fn __len__(&self) -> usize {
        self.inner.len()
    }


    fn snapshot(&self) -> PyResult<Vec<u8>> {
        serde_json::to_vec(&self.inner.snapshot())
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {e}")))
    }


    fn merge(&mut self, data: &[u8]) -> PyResult<i64> {
        const MAX_PAYLOAD_BYTES: usize = 1_048_576; // 1 MB
        if data.len() > MAX_PAYLOAD_BYTES {
            return Err(PyValueError::new_err(format!(
                "Payload too large: {} bytes (max {})",
                data.len(),
                MAX_PAYLOAD_BYTES
            )));
        }
        let remote: or_map::OrMapSnapshot = serde_json::from_slice(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {e}")))?;
        Ok(self.inner.merge(&remote))
    }


    #[getter]
    fn state_total(&self) -> i64 {
        self.inner.state_total()
    }


    #[getter]
    fn tombstone_count(&self) -> usize {
        self.inner.tombstone_count()
    }

    fn __repr__(&self) -> String {
        format!("StateMap(keys={})", self.inner.len())
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// GossipCore — Opaque gossip scheduler (EntropyAgent + StateMap combined)
//
// This is the primary interface for the SDK. It hides all adaptive scheduling
// logic behind prepare_round() / report_delivery() / merge_remote().
// Python never sees get_k(), state_total, entropy estimates, or sleep logic.
// ═══════════════════════════════════════════════════════════════════════════

#[pyclass(name = "GossipDecision")]
#[derive(Clone)]
struct PyGossipDecision {
    #[pyo3(get)]
    k: u32,
    #[pyo3(get)]
    snapshot: Option<Vec<u8>>,
    #[pyo3(get)]
    sleep_ms: u64,
}

#[pymethods]
impl PyGossipDecision {
    fn __repr__(&self) -> String {
        format!(
            "GossipDecision(k={}, snapshot={}B)",
            self.k,
            self.snapshot.as_ref().map_or(0, |s| s.len()),
        )
    }
}


#[pyclass(name = "GossipCore")]
struct PyGossipCore {
    agent: entropy_agent::EntropyAgent,
    state: or_map::OrMap,
    gossip_rounds: u64,
    gossip_interval_ms: f64,
}

#[pymethods]
impl PyGossipCore {
    #[new]
    #[pyo3(signature = (node_id, seed = 0, gossip_interval_ms = 500.0))]
    fn new(node_id: u64, seed: u64, gossip_interval_ms: f64) -> Self {
        Self {
            agent: entropy_agent::EntropyAgent::new(node_id, seed),
            state: or_map::OrMap::new(node_id),
            gossip_rounds: 0,
            gossip_interval_ms,
        }
    }

    // ─── State API (delegates to OrMap) ──────────────────────────────

    fn set(&mut self, key: &str, value: &str) {
        self.state.set(key, value);
    }

    fn get(&self, key: &str) -> Option<String> {
        self.state.get(key).map(|s| s.to_string())
    }

    fn get_conflicts(&self, key: &str) -> Vec<String> {
        self.state.get_all(key).iter().map(|s| s.to_string()).collect()
    }

    fn delete(&mut self, key: &str) {
        self.state.delete(key);
    }

    fn keys(&self) -> Vec<String> {
        self.state.keys().iter().map(|s| s.to_string()).collect()
    }

    fn __len__(&self) -> usize {
        self.state.len()
    }

    // ─── Gossip Decision (opaque) ────────────────────────────────────

    fn prepare_round(&mut self, current_time_ms: f64) -> PyResult<PyGossipDecision> {
        self.gossip_rounds += 1;
        let state_total = self.state.state_total();
        let k = self.agent.get_k(current_time_ms, state_total);

        if k > 0 {
            let snapshot = serde_json::to_vec(&self.state.snapshot())
                .map_err(|e| PyValueError::new_err(format!("Serialization failed: {e}")))?;
            Ok(PyGossipDecision {
                k,
                snapshot: Some(snapshot),
                sleep_ms: 0,
            })
        } else {
            Ok(PyGossipDecision {
                k: 0,
                snapshot: None,
                sleep_ms: 0,
            })
        }
    }

    fn report_delivery(&mut self, delivered: bool) {
        self.agent.on_packet_sent(delivered);
    }

    fn merge_remote(&mut self, data: &[u8], current_time_ms: f64) -> PyResult<i64> {
        const MAX_PAYLOAD_BYTES: usize = 1_048_576; // 1 MB
        if data.len() > MAX_PAYLOAD_BYTES {
            return Err(PyValueError::new_err(format!(
                "Payload too large: {} bytes (max {})",
                data.len(),
                MAX_PAYLOAD_BYTES
            )));
        }
        let remote: or_map::OrMapSnapshot = serde_json::from_slice(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {e}")))?;
        let delta = self.state.merge(&remote);
        self.agent.on_receive(delta, current_time_ms);
        Ok(delta)
    }

    // ─── Snapshot (for direct merge testing, not used in gossip loop) ─

    fn snapshot(&self) -> PyResult<Vec<u8>> {
        serde_json::to_vec(&self.state.snapshot())
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {e}")))
    }

    // ─── Introspection (minimal — for dashboard/monitoring only) ─────

    #[getter]
    fn key_count(&self) -> usize {
        self.state.len()
    }

    #[getter]
    fn tombstone_count(&self) -> usize {
        self.state.tombstone_count()
    }

    #[getter]
    fn rounds(&self) -> u64 {
        self.gossip_rounds
    }

    fn get_stats(&self) -> PyResult<std::collections::HashMap<String, f64>> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("key_count".to_string(), self.state.len() as f64);
        stats.insert("tombstone_count".to_string(), self.state.tombstone_count() as f64);
        stats.insert("gossip_rounds".to_string(), self.gossip_rounds as f64);
        stats.insert("convergence_ms".to_string(),
            self.agent.estimate_convergence_ms(self.gossip_interval_ms));
        stats.insert("is_converged".to_string(),
            if self.agent.entropy_estimate() < 0.08 { 1.0 } else { 0.0 });
        Ok(stats)
    }

    fn __repr__(&self) -> String {
        format!(
            "GossipCore(keys={}, rounds={})",
            self.state.len(),
            self.gossip_rounds,
        )
    }
}


#[pymodule]
fn entropy_state_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEntropyAgent>()?;
    m.add_class::<PyStateMap>()?;
    m.add_class::<PyGossipCore>()?;
    m.add_class::<PyGossipDecision>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

