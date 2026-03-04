"""Q.ANT target extension for the docc compiler."""

from typing import Callable, Optional, Dict, Any
from ._qant import register_plugin_qant, schedule_qant, compile_qant


def _schedule(sdfg, category: str, kwargs: Dict[str, Any]) -> None:
    schedule_qant(sdfg._ptr, category)


def _compile(
    sdfg, out_dir: str, inst_mode: str, capture: bool, kwargs: Dict[str, Any]
) -> str:
    return compile_qant(sdfg._ptr, out_dir, "qant", inst_mode, capture)


def register_docc_plugin():
    from docc.sdfg import _plugin_context

    register_plugin_qant(_plugin_context())

    from docc.python import register_target_overrides

    register_target_overrides("qant", _schedule, _compile)
