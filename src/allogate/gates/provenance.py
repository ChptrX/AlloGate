"""Explicit acyclic provenance for multiscale Gate interpretation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from allogate.config.enums import StrEnum

from allogate.config.hashing import stable_digest

from .identities import EntityRef


class ProvenanceRelation(StrEnum):
    MEMBER_OF = "member_of"
    ENDPOINT_OF = "endpoint_of"
    INDUCES = "induces"
    AGGREGATES_INTO = "aggregates_into"
    MESSAGES_TO = "messages_to"
    READ_BY = "read_by"


@dataclass(frozen=True, order=True, slots=True)
class ProvenanceEdge:
    source: str
    target: str
    relation: ProvenanceRelation

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("a provenance edge cannot be a self-loop")


@dataclass(frozen=True, slots=True)
class ProvenanceDAG:
    nodes: tuple[EntityRef, ...]
    edges: tuple[ProvenanceEdge, ...]
    schema_version: str = "allogate.provenance.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "allogate.provenance.v1":
            raise ValueError(f"unsupported provenance schema: {self.schema_version}")
        ordered_nodes = tuple(sorted(self.nodes, key=lambda node: node.uid))
        ordered_edges = tuple(sorted(self.edges))
        node_uids = [node.uid for node in ordered_nodes]
        if len(set(node_uids)) != len(node_uids):
            raise ValueError("duplicate provenance node")
        if len(set(ordered_edges)) != len(ordered_edges):
            raise ValueError("duplicate provenance edge")
        known = set(node_uids)
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = dict.fromkeys(node_uids, 0)
        for edge in ordered_edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("a provenance edge references an unknown node")
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
        queue = deque(sorted(uid for uid, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in sorted(outgoing[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(node_uids):
            raise ValueError("provenance graph contains a cycle")
        object.__setattr__(self, "nodes", ordered_nodes)
        object.__setattr__(self, "edges", ordered_edges)

    @classmethod
    def build(
        cls,
        nodes: Iterable[EntityRef],
        edges: Iterable[ProvenanceEdge],
    ) -> "ProvenanceDAG":
        return cls(tuple(nodes), tuple(edges))

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "nodes": [node.uid for node in self.nodes],
                "edges": [
                    {"source": edge.source, "target": edge.target, "relation": edge.relation.value}
                    for edge in self.edges
                ],
            }
        )

    def _closure(self, node_uid: str, *, reverse: bool) -> tuple[str, ...]:
        known = {node.uid for node in self.nodes}
        if node_uid not in known:
            raise KeyError(node_uid)
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            source, target = (edge.target, edge.source) if reverse else (edge.source, edge.target)
            adjacency[source].append(target)
        seen: set[str] = set()
        stack = list(adjacency[node_uid])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current])
        return tuple(sorted(seen))

    def descendants(self, node_uid: str) -> tuple[str, ...]:
        return self._closure(node_uid, reverse=False)

    def ancestors(self, node_uid: str) -> tuple[str, ...]:
        return self._closure(node_uid, reverse=True)

    def validate_registry(self, registry: "object") -> None:
        from .registry import GateRegistry

        if not isinstance(registry, GateRegistry):
            raise TypeError("registry must be a GateRegistry")
        known = {node.uid for node in self.nodes}
        for gate in registry.gates:
            missing = set(gate.provenance_nodes).difference(known)
            if missing:
                raise ValueError(f"Gate {gate.uid} references unknown provenance nodes: {sorted(missing)}")
