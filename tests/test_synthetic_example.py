from pathlib import Path
import runpy

import numpy as np


def test_synthetic_example_runs_end_to_end(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "synthetic_gate_audit.py")
    )
    result = namespace["run_example"](tmp_path / "artifacts")
    assert result["schema_version"] == "allogate.synthetic_example.v1"
    assert len(result["gate_uids"]) == 3
    assert np.asarray(result["jacobian"]).shape == (3, 3)
    assert result["hessian_maximum_asymmetry"] <= 1.0e-12
    assert [point["dose"] for point in result["relay_dose_response"]] == [1.0, 0.5, 0.0]
    assert len(result["artifact_digest"]) == 64
    assert (tmp_path / "artifacts" / "refs" / "examples" / "synthetic-gate-audit.json").is_file()
