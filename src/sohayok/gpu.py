"""Let Sohayok's short GPU bursts (speech recognition) go ahead of other GPU programs.

With a training job keeping the GPU at 100%, recognising a 2 s command took 18 s at normal
priority, 13 s at high, and 3.4 s at real-time (RTX 4060, Windows 11). The priority is raised
only while Sohayok is recognising speech and restored right after, so other programs lose only
those bursts.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import threading
from collections.abc import Iterator

log = logging.getLogger(__name__)

# D3DKMT_SCHEDULINGPRIORITYCLASS
CLASSES = {"idle": 0, "below_normal": 1, "normal": 2, "above_normal": 3, "high": 4, "realtime": 5}
_lock = threading.Lock()
_depth = 0


def set_priority(level: str) -> bool:
    """Set this process's GPU scheduling priority class; False if Windows refused."""
    try:
        process = ctypes.windll.kernel32.GetCurrentProcess()
        status = ctypes.WinDLL("gdi32").D3DKMTSetProcessSchedulingPriorityClass(
            ctypes.c_void_p(process), ctypes.c_int(CLASSES[level])
        )
    except (AttributeError, OSError, KeyError):
        return False
    if status != 0:
        log.debug("GPU priority %s refused (0x%x)", level, status & 0xFFFFFFFF)
    return status == 0


@contextlib.contextmanager
def priority(level: str = "realtime") -> Iterator[None]:
    """Raise the GPU priority for the duration of the block (nested blocks are fine)."""
    global _depth
    if level not in CLASSES or level == "normal":
        yield
        return
    with _lock:
        _depth += 1
        if _depth == 1:
            set_priority(level)
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                set_priority("normal")
