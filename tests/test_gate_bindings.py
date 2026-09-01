import pytest


torch = pytest.importorskip("torch")

from allogate.gates.bindings import ModelGateBinding  # noqa: E402
from allogate.gates.identities import EntityLevel, EntityRef, GateSpec  # noqa: E402
from allogate.gates.registry import GateRegistry  # noqa: E402


def registry_and_gates():
    residues = tuple(
        EntityRef("example", EntityLevel.RESIDUE, f"residue-{index:04d}") for index in range(2)
    )
    locals_ = tuple(
        EntityRef("example", EntityLevel.LOCAL_ELEMENT, f"local-{index:04d}") for index in range(2)
    )
    contact_entity = EntityRef("example", EntityLevel.CONTACT, "contact-0000-0001")
    state = tuple(GateSpec.state(item) for item in residues)
    relay = tuple(GateSpec.relay(item) for item in locals_)
    contact = GateSpec.contact(residues[0], residues[1], provenance_node=contact_entity.uid)
    registry = GateRegistry.from_specs("example", (*state, *relay, contact))
    return registry, state, relay, contact


def test_runtime_binding_repeats_semantic_uids_across_graphs() -> None:
    registry, state, relay, contact = registry_and_gates()
    binding = ModelGateBinding.build(
        registry,
        {
            "residue_state": (state[0].uid, state[1].uid, state[0].uid, state[1].uid),
            "local_to_unit": (relay[0].uid, relay[1].uid, relay[0].uid, relay[1].uid),
            "residue_edges": (contact.uid, contact.uid),
        },
    )
    values = torch.linspace(0.2, 0.8, len(registry.layout), requires_grad=True)
    controls = binding.bind(registry, values)
    assert controls.residue_state.shape == (4,)
    assert controls.residue_state[0] == controls.residue_state[2]
    assert controls.residue_edges.shape == (2,)
    controls.residue_state.sum().backward()
    assert values.grad is not None


def test_incompatible_family_is_rejected() -> None:
    registry, _, _, contact = registry_and_gates()
    with pytest.raises(ValueError, match="incompatible"):
        ModelGateBinding.build(registry, {"residue_state": (contact.uid,)})
