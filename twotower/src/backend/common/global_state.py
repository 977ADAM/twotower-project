import gc
import threading

GLOBAL_STATE_TRACKER = threading.local()
GLOBAL_SETTINGS_TRACKER = threading.local()

def set_global_attribute(name, value):
    setattr(GLOBAL_STATE_TRACKER, name, value)

def get_global_attribute(name, default=None, set_to_default=False):
    attr = getattr(GLOBAL_STATE_TRACKER, name, None)
    if attr is None and default is not None:
        attr = default
        if set_to_default:
            set_global_attribute(name, attr)
    return attr