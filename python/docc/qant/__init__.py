"""Q.ANT target extension for the docc compiler."""

from typing import Callable, Optional, Dict, Any
from ._qant import (
    register_plugin_qant,
)


def register_docc_plugin():
    from docc.sdfg import _plugin_context

    register_plugin_qant(_plugin_context())
