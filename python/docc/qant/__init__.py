"""Q.ANT target extension for the docc compiler."""

from typing import Callable, Optional, Dict, Any
from ._qant import (
    register_plugin_qant,
    schedule_qant,
    compile_qant,
    expand_qant,
    qant_compile_hook,
)


def _schedule(sdfg, category: str, kwargs: Dict[str, Any]) -> None:
    schedule_qant(sdfg._ptr, category)


def _compile(
    sdfg, out_dir: str, inst_mode: str, capture: bool, kwargs: Dict[str, Any]
) -> str:
    qant_compile_hook(sdfg._ptr, out_dir, "qant", inst_mode, capture)
    # default compile as if we did not even hook it. We only do that for the offload-analysis
    return sdfg._compile(
        out_dir, "qant", inst_mode, capture, kwargs.get("debug_build", False)
    )


def _expand(sdfg, category: str, kwargs: Dict[str, Any]) -> None:
    expand_qant(sdfg._ptr, category)
    sdfg.expand()


def register_docc_plugin():
    from docc.sdfg import _plugin_context

    register_plugin_qant(_plugin_context())

    from docc.python import register_target_overrides

    register_target_overrides("qant", _schedule, _compile, _expand)
