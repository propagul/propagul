"""Discovery service — resolves available gossip peers."""

import logging
from typing import List, Optional
from propagul.types import PeerAddress

logger = logging.getLogger("propagul.discovery")


class StaticDiscovery:
    """Static peer list discovery — for local/dev setups.

    Usage:
        discovery = StaticDiscovery([
            PeerAddress("127.0.0.1", 9001),
            PeerAddress("127.0.0.1", 9002),
        ])
    """

    def __init__(self, peers: Optional[List[PeerAddress]] = None):
        self._peers: List[PeerAddress] = list(peers) if peers else []

    def add_peer(self, peer: PeerAddress) -> None:
        """Add a peer to the list (idempotent)."""
        if peer not in self._peers:
            self._peers.append(peer)
            logger.debug("Added peer: %s", peer)

    def remove_peer(self, peer: PeerAddress) -> None:
        """Remove a peer from the list."""
        if peer in self._peers:
            self._peers.remove(peer)
            logger.debug("Removed peer: %s", peer)

    def get_peers(self, exclude: Optional[PeerAddress] = None) -> List[PeerAddress]:
        """Return all known peers, optionally excluding self."""
        if exclude is None:
            return list(self._peers)
        return [p for p in self._peers if p != exclude]

    @property
    def peer_count(self) -> int:
        return len(self._peers)
