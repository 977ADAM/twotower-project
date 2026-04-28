from __future__ import annotations

import os
import types
from functools import wraps

_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _is_internal_frame(frame: types.FrameType) -> bool:
    return os.path.normpath(frame.f_code.co_filename).startswith(_SRC_ROOT)


def _process_traceback_frames(tb: types.TracebackType | None) -> types.TracebackType | None:
    frames: list[types.TracebackType] = []
    while tb is not None:
        if not _is_internal_frame(tb.tb_frame):
            frames.append(tb)
        tb = tb.tb_next
    result: types.TracebackType | None = None
    for frame_tb in reversed(frames):
        result = types.TracebackType(result, frame_tb.tb_frame, frame_tb.tb_lasti, frame_tb.tb_lineno)
    return result


def filter_traceback(fn):
    @wraps(fn)
    def error_handler(*args, **kwargs):
        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb
    return error_handler
