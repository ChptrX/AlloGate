import pytest

from allogate.evaluation import DoseSchedule


def test_dose_schedule_is_deterministic_and_single_gate() -> None:
    schedule = DoseSchedule(("gate-a", "gate-b"), (0.0, 1.0, 0.5, 0.5))
    assert schedule.doses == (1.0, 0.5, 0.0)
    requests = schedule.requests("1" * 64)
    assert len(requests) == 6
    assert all(len(request.overrides) == 1 for request in requests)
    assert requests[0].overrides == (("gate-a", 1.0),)


def test_dose_schedule_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DoseSchedule(("gate-a",), (1.2,))
