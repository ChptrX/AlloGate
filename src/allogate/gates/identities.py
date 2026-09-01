"""Stable, display-independent identities for public Gate definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from allogate.config.hashing import stable_digest


_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class EntityLevel(StrEnum):
    RESIDUE = "residue"
    CONTACT = "contact"
    LOCAL_ELEMENT = "local_element"
    INTERFACE = "interface"
    STRUCTURAL_UNIT = "structural_unit"
    RELATION = "relation"
    READOUT = "readout"


class GateFamily(StrEnum):
    STATE = "state"
    RELAY = "relay"
    CONTACT = "contact"


@dataclass(frozen=True, order=True, slots=True)
class EntityRef:
    """A semantic entity identity independent of runtime numbering."""

    namespace: str
    level: EntityLevel
    key: str

    def __post_init__(self) -> None:
        if _TOKEN.fullmatch(self.namespace) is None:
            raise ValueError(f"invalid entity namespace: {self.namespace!r}")
        if _TOKEN.fullmatch(self.key) is None:
            raise ValueError(f"invalid entity key: {self.key!r}")

    @property
    def uid(self) -> str:
        return f"ag-entity:v1:{self.namespace}:{self.level.value}:{self.key}"


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One canonical Gate definition.

    Display names and aliases are intentionally excluded from the UID. Contact
    endpoint order is canonicalized, making the identity direction independent.
    """

    family: GateFamily
    targets: tuple[EntityRef, ...]
    provenance_nodes: tuple[str, ...]
    relation: str = "feature"
    display_name: str = ""
    aliases: tuple[str, ...] = ()
    default: float = 1.0

    def __post_init__(self) -> None:
        if self.default != 1.0:
            raise ValueError("the immutable no-intervention Gate default must be exactly one")
        if _TOKEN.fullmatch(self.relation) is None:
            raise ValueError(f"invalid Gate relation: {self.relation!r}")
        expected_targets = 2 if self.family is GateFamily.CONTACT else 1
        if len(self.targets) != expected_targets:
            raise ValueError(f"{self.family.value} Gate requires {expected_targets} target(s)")
        namespaces = {target.namespace for target in self.targets}
        if len(namespaces) != 1:
            raise ValueError("all Gate targets must share one namespace")
        canonical_targets = tuple(sorted(self.targets, key=lambda target: target.uid))
        if len({target.uid for target in canonical_targets}) != len(canonical_targets):
            raise ValueError("Gate targets must be distinct")
        if self.family is GateFamily.CONTACT and any(
            target.level is EntityLevel.READOUT for target in canonical_targets
        ):
            raise ValueError("a Contact Gate cannot target a readout")
        if self.family is GateFamily.CONTACT and len({target.level for target in canonical_targets}) != 1:
            raise ValueError("Contact Gate endpoints must be at the same entity level")
        object.__setattr__(self, "targets", canonical_targets)
        if not self.provenance_nodes or len(set(self.provenance_nodes)) != len(self.provenance_nodes):
            raise ValueError("provenance_nodes must be non-empty and unique")
        normalized_aliases = tuple(sorted(self.aliases))
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("Gate aliases must be unique")
        for alias in normalized_aliases:
            if _TOKEN.fullmatch(alias) is None:
                raise ValueError(f"invalid Gate alias: {alias!r}")
        object.__setattr__(self, "aliases", normalized_aliases)

    @property
    def namespace(self) -> str:
        return self.targets[0].namespace

    @property
    def level(self) -> EntityLevel:
        return EntityLevel.CONTACT if self.family is GateFamily.CONTACT else self.targets[0].level

    @property
    def intervention_semantics(self) -> str:
        return {
            GateFamily.STATE: "state_replace",
            GateFamily.RELAY: "relay_scale",
            GateFamily.CONTACT: "contact_scale",
        }[self.family]

    @property
    def uid(self) -> str:
        semantic = {
            "schema": "allogate.gate.v1",
            "namespace": self.namespace,
            "family": self.family.value,
            "targets": [target.uid for target in self.targets],
            "relation": self.relation,
            "semantics": self.intervention_semantics,
        }
        digest = stable_digest(semantic)[:24]
        return f"ag-gate:v1:{self.namespace}:{self.family.value}:{self.level.value}:{digest}"

    @classmethod
    def state(
        cls,
        target: EntityRef,
        *,
        provenance_node: str | None = None,
        display_name: str = "",
        aliases: tuple[str, ...] = (),
    ) -> "GateSpec":
        return cls(
            family=GateFamily.STATE,
            targets=(target,),
            provenance_nodes=(provenance_node or target.uid,),
            relation="stored_state",
            display_name=display_name,
            aliases=aliases,
        )

    @classmethod
    def relay(
        cls,
        target: EntityRef,
        *,
        provenance_node: str | None = None,
        relation: str = "parent_relay",
        display_name: str = "",
        aliases: tuple[str, ...] = (),
    ) -> "GateSpec":
        return cls(
            family=GateFamily.RELAY,
            targets=(target,),
            provenance_nodes=(provenance_node or target.uid,),
            relation=relation,
            display_name=display_name,
            aliases=aliases,
        )

    @classmethod
    def contact(
        cls,
        first: EntityRef,
        second: EntityRef,
        *,
        provenance_node: str,
        display_name: str = "",
        aliases: tuple[str, ...] = (),
    ) -> "GateSpec":
        return cls(
            family=GateFamily.CONTACT,
            targets=(first, second),
            provenance_nodes=(provenance_node,),
            relation="spatial_contact",
            display_name=display_name,
            aliases=aliases,
        )
