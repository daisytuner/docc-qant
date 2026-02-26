"""Q.ANT target extension for the docc compiler."""

from ._qant import register_plugin_qant, schedule_qant


def _schedule(sdfg, category: str) -> None:
    schedule_qant(sdfg._ptr, category)


def register_docc_plugin():
    from docc.sdfg import _plugin_context

    register_plugin_qant(_plugin_context())

    from docc.python import register_target

    register_target("qant", _schedule)
