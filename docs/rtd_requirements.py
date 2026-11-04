"""Print the package's pinned dependencies as requirements files for a CPU-only docs build.

Read the Docs builds on a CPU machine with a 15-minute limit, and the docs run no notebooks, so the ~3 GB of
nvidia-* and triton wheels are left out and torch is the CPU build (see .readthedocs.yaml):

    python docs/rtd_requirements.py torch   # torch, torchvision, torchaudio pinned to +cpu, for the PyTorch CPU index
    python docs/rtd_requirements.py         # every other pin, for PyPI

Every other pin is taken from pyproject.toml as it is, so the API pages are built against the same versions as a
real install.
"""
import pathlib
import re
import sys
import tomllib

SKIP = re.compile(r"^(python|nvidia-.*|triton)$")
TORCH = {"torch", "torchvision", "torchaudio"}


def requirement(name: str, spec: object) -> str:
    """One pyproject spec as a pip requirement; fails loudly on anything it does not know how to translate."""
    if isinstance(spec, str):
        if re.fullmatch(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", spec):  # an exact version
            return f"{name}=={spec}"
        caret = re.fullmatch(r"\^(\d+(?:\.\d+)*)", spec)
        if caret:  # poetry caret: >=x.y.z, < next major (or next minor/patch while the leading parts are 0)
            parts = [int(p) for p in caret[1].split(".")]
            i = next((k for k, p in enumerate(parts) if p != 0), len(parts) - 1)
            return f"{name}>={caret[1]},<{'.'.join(map(str, parts[:i] + [parts[i] + 1]))}"
    raise SystemExit(f"docs/rtd_requirements.py: cannot translate the pyproject spec {name} = {spec!r}; "
                     "add it to requirement()")


deps = tomllib.loads((pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())["tool"]["poetry"]["dependencies"]
want_torch = sys.argv[1:] == ["torch"]
for name, spec in deps.items():
    if SKIP.match(name) or (name in TORCH) != want_torch:
        continue
    req = requirement(name, spec)
    print(req + "+cpu" if want_torch else req)
