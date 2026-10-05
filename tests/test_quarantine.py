"""wp_baseline stays out of the market decision path.

It is the benchmark, and the original-nfl4th fallback for plays the market engine can't
decide. Only `fallback.py` may import it, and `decision.py` loads `fallback` lazily, so
importing any production module never loads the WP models.
"""

import ast
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "nfl4th_py"
BASELINE = "nfl4th_py.models.wp_baseline"
# modules allowed to import the benchmark (benchmark/validation tooling only)
ALLOWED = {PKG / "models" / "wp_baseline.py", PKG / "fallback.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_no_production_module_imports_wp_baseline():
    offenders = [
        str(p.relative_to(PKG))
        for p in PKG.rglob("*.py")
        if p not in ALLOWED and any(n == BASELINE or n.endswith("wp_baseline") for n in _imports(p))
    ]
    assert not offenders, f"production modules import wp_baseline: {offenders}"


def test_importing_production_modules_does_not_load_wp_baseline():
    modules = [
        p.relative_to(PKG.parent).with_suffix("").as_posix().replace("/", ".")
        for p in PKG.rglob("*.py")
        if p not in ALLOWED and p.name != "__main__.py" and p.name != "fallback.py"
    ]
    modules = [m.removesuffix(".__init__") for m in modules]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        f"assert {BASELINE!r} not in sys.modules, 'wp_baseline was loaded'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
