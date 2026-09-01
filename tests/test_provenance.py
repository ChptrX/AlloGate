import pytest

from allogate.gates.identities import EntityLevel, EntityRef, GateSpec
from allogate.gates.provenance import ProvenanceDAG, ProvenanceEdge, ProvenanceRelation
from allogate.gates.registry import GateRegistry


def graph_objects():
    residue_a = EntityRef("example", EntityLevel.RESIDUE, "residue-a-0001")
    residue_b = EntityRef("example", EntityLevel.RESIDUE, "residue-a-0002")
    contact = EntityRef("example", EntityLevel.CONTACT, "contact-0001-0002")
    local = EntityRef("example", EntityLevel.LOCAL_ELEMENT, "local-0001")
    unit = EntityRef("example", EntityLevel.STRUCTURAL_UNIT, "unit-0001")
    readout = EntityRef("example", EntityLevel.READOUT, "route-contact")
    nodes = (residue_a, residue_b, contact, local, unit, readout)
    edges = (
        ProvenanceEdge(residue_a.uid, contact.uid, ProvenanceRelation.ENDPOINT_OF),
        ProvenanceEdge(residue_b.uid, contact.uid, ProvenanceRelation.ENDPOINT_OF),
        ProvenanceEdge(residue_a.uid, local.uid, ProvenanceRelation.MEMBER_OF),
        ProvenanceEdge(residue_b.uid, local.uid, ProvenanceRelation.MEMBER_OF),
        ProvenanceEdge(contact.uid, local.uid, ProvenanceRelation.INDUCES),
        ProvenanceEdge(local.uid, unit.uid, ProvenanceRelation.MEMBER_OF),
        ProvenanceEdge(unit.uid, readout.uid, ProvenanceRelation.READ_BY),
    )
    return nodes, edges


def test_provenance_closure_and_digest_are_deterministic() -> None:
    nodes, edges = graph_objects()
    left = ProvenanceDAG.build(nodes, edges)
    right = ProvenanceDAG.build(reversed(nodes), reversed(edges))
    assert left.digest == right.digest
    residue = next(node for node in nodes if node.key == "residue-a-0001")
    readout = next(node for node in nodes if node.level is EntityLevel.READOUT)
    assert readout.uid in left.descendants(residue.uid)
    assert residue.uid in left.ancestors(readout.uid)


def test_provenance_cycle_is_rejected() -> None:
    nodes, edges = graph_objects()
    residue = next(node for node in nodes if node.level is EntityLevel.RESIDUE)
    readout = next(node for node in nodes if node.level is EntityLevel.READOUT)
    with pytest.raises(ValueError, match="cycle"):
        ProvenanceDAG.build(
            nodes,
            (*edges, ProvenanceEdge(readout.uid, residue.uid, ProvenanceRelation.INDUCES)),
        )


def test_registry_references_must_exist_in_provenance() -> None:
    nodes, edges = graph_objects()
    dag = ProvenanceDAG.build(nodes, edges)
    residue_a, residue_b = nodes[0], nodes[1]
    contact = next(node for node in nodes if node.level is EntityLevel.CONTACT)
    registry = GateRegistry.from_specs(
        "example",
        (
            GateSpec.state(residue_a),
            GateSpec.contact(residue_a, residue_b, provenance_node=contact.uid),
        ),
    )
    dag.validate_registry(registry)
    missing = GateSpec.relay(
        EntityRef("example", EntityLevel.LOCAL_ELEMENT, "missing"),
    )
    with pytest.raises(ValueError, match="unknown provenance"):
        dag.validate_registry(GateRegistry.from_specs("example", (missing,)))

