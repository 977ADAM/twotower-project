import inspect
import os
import traceback
import types
from functools import wraps

# from twotower.src import backend
# from twotower.src import tree
from twotower.src.api_export import twotower_export
from twotower.src.backend.common import global_state

@twotower_export("twotower.config.is_traceback_filtering_enabled")
def is_traceback_filtering_enabled():
    return global_state.get_global_attribute("traceback_filtering", True)

def filter_traceback(fn):
    """Filter out Keras-internal traceback frames in exceptions raised by fn."""

    @wraps(fn)
    def error_handler(*args, **kwargs):
        if not is_traceback_filtering_enabled():
            return fn(*args, **kwargs)

        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            # To get the full stack trace, call:
            # `keras.config.disable_traceback_filtering()`
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb

    return error_handler