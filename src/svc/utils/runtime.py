from __future__ import annotations

import importlib.util
import platform
import sys


def check_python_version() -> None:
    if sys.version_info >= (3, 11):
        return
    raise SystemExit(
        "This project requires Python 3.11+.\n"
        f"Current Python: {sys.version.split()[0]}\n"
        "Run: source .venv/bin/activate"
    )


def check_pitch_dependency(pitch_name: str, backend: str) -> None:
    if pitch_name.replace("-", "_").lower() != "rmvpe":
        return
    backend = resolve_rmvpe_backend(backend)
    module = "mlx_rmvpe" if backend == "mlx" else "rmvpe_onnx"
    if importlib.util.find_spec(module) is not None:
        return
    raise SystemExit(
        f"RMVPE {backend} backend is selected but {module} is not installed.\n"
        f"Install with: python -m pip install -e '.[rmvpe-{backend}]'\n"
        "Or set pitch.name: pyin in the config."
    )


def resolve_rmvpe_backend(backend: str) -> str:
    normalized = backend.replace("-", "_").lower()
    if normalized != "auto":
        return normalized
    return "mlx" if platform.system() == "Darwin" else "onnx"
