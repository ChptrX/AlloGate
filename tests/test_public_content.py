from pathlib import Path
import runpy


def test_public_tree_passes_content_gate() -> None:
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "public_content_audit.py"))
    assert namespace["audit"]() == []

