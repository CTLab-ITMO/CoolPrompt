"""Loader for the byte-identical vendored RIDER Genesis source."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict

from coolprompt.optimizer.rider import _llm_shim


VENDORED_RIDER_DIR = Path(__file__).resolve().parent / "vendor" / "rider"
_ASSISTANT_MODULE_NAME = "coolprompt.optimizer.rider._vendored_assistant"
_MISSING = object()


def _install_temporary_rider_modules() -> Dict[str, Any]:
    """Install temporary ``rider`` modules for loading the vendored assistant.

    Returns:
        Previous ``sys.modules`` entries so they can be restored after import.
    """

    previous: Dict[str, Any] = {}
    for name in ("rider", "rider.llm", "rider.llm.client"):
        previous[name] = sys.modules.get(name, _MISSING)

    rider_pkg = types.ModuleType("rider")
    rider_pkg.__path__ = [str(VENDORED_RIDER_DIR)]

    llm_pkg = types.ModuleType("rider.llm")
    llm_pkg.__path__ = [str(VENDORED_RIDER_DIR / "llm")]

    sys.modules["rider"] = rider_pkg
    sys.modules["rider.llm"] = llm_pkg
    sys.modules["rider.llm.client"] = _llm_shim
    return previous


def _restore_modules(previous: Dict[str, Any]) -> None:
    """Restore ``sys.modules`` entries captured before vendored import.

    Args:
        previous: Mapping returned by ``_install_temporary_rider_modules``.
    """

    for name, module in previous.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _disable_instructor_client(self: Any, model: str) -> None:
    """Disable vendored structured-output client initialization.

    Args:
        self: Vendored ``RiderGenesis`` instance.
        model: Model name requested by the vendored runtime.

    Returns:
        ``None`` so RIDER falls back to the LangChain shim path.
    """

    _ = (self, model)
    return None


def load_rider_genesis() -> type:
    """Load ``RiderGenesis`` from the vendored assistant.py without editing it.

    Returns:
        Vendored ``RiderGenesis`` class with CoolPrompt runtime patches applied.

    Raises:
        ImportError: If the vendored assistant module cannot be loaded.
    """

    if _ASSISTANT_MODULE_NAME in sys.modules:
        return sys.modules[_ASSISTANT_MODULE_NAME].RiderGenesis

    assistant_path = VENDORED_RIDER_DIR / "assistant.py"
    spec = importlib.util.spec_from_file_location(
        _ASSISTANT_MODULE_NAME,
        assistant_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load vendored RIDER assistant from {assistant_path}"
        )

    module = importlib.util.module_from_spec(spec)
    previous = _install_temporary_rider_modules()
    try:
        sys.modules[_ASSISTANT_MODULE_NAME] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_ASSISTANT_MODULE_NAME, None)
        raise
    finally:
        _restore_modules(previous)

    # Keep structured-output calls on the same LangChain path as normal calls.
    module.RiderGenesis._instructor_client = _disable_instructor_client
    return module.RiderGenesis
