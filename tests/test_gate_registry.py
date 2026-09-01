import pytest

from allogate.gates.identities import EntityLevel, EntityRef, GateFamily, GateSpec
from allogate.gates.registry import GateRegistry


def entities():
    residue_a = EntityRef("example", EntityLevel.RESIDUE, "chain-a-residue-0001")
    residue_b = EntityRef("example", EntityLevel.RESIDUE, "chain-a-residue-0002")
    local = EntityRef("example", EntityLevel.LOCAL_ELEMENT, "local-0001")
    contact = EntityRef("example", EntityLevel.CONTACT, "contact-0001-0002")
    return residue_a, residue_b, local, contact


def specs():
    residue_a, residue_b, local, contact = entities()
    return (
        GateSpec.state(residue_a, aliases=("state-a",)),
        GateSpec.relay(local, aliases=("relay-local",)),
        GateSpec.contact(residue_a, residue_b, provenance_node=contact.uid, aliases=("contact-ab",)),
    )


def test_uid_ignores_display_text_and_contact_direction() -> None:
    residue_a, residue_b, _, contact = entities()
    first = GateSpec.contact(residue_a, residue_b, provenance_node=contact.uid, display_name="first")
    reversed_gate = GateSpec.contact(residue_b, residue_a, provenance_node=contact.uid, display_name="second")
    assert first.uid == reversed_gate.uid
    assert GateSpec.state(residue_a, display_name="one").uid == GateSpec.state(
        residue_a, display_name="two"
    ).uid


def test_registry_and_layout_are_independent_of_input_order() -> None:
    gates = specs()
    left = GateRegistry.from_specs("example", gates)
    right = GateRegistry.from_specs("example", reversed(gates))
    assert left.gates == right.gates
    assert left.digest == right.digest
    assert left.layout == right.layout
    assert left.layout.digest == right.layout.digest
    assert [entry.global_index for entry in left.layout.entries] == list(range(len(gates)))
    assert left.resolve("contact-ab").family is GateFamily.CONTACT


def test_layout_vector_is_all_one_unless_explicitly_overridden() -> None:
    registry = GateRegistry.from_specs("example", specs())
    target = registry.resolve("relay-local")
    vector = registry.layout.vector({target.uid: 0.25})
    assert vector[registry.layout.index(target.uid)] == 0.25
    assert sum(value == 1.0 for value in vector) == len(vector) - 1


def test_alias_collisions_are_rejected() -> None:
    residue_a, _, local, _ = entities()
    with pytest.raises(ValueError, match="ambiguous"):
        GateRegistry.from_specs(
            "example",
            (
                GateSpec.state(residue_a, aliases=("same",)),
                GateSpec.relay(local, aliases=("same",)),
            ),
        )


def test_non_one_default_is_rejected() -> None:
    residue_a, _, _, _ = entities()
    with pytest.raises(ValueError, match="exactly one"):
        GateSpec(
            family=GateFamily.STATE,
            targets=(residue_a,),
            provenance_nodes=(residue_a.uid,),
            default=0.5,
        )

