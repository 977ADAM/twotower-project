try:
    import namex
except ImportError:
    namex = None

REGISTERED_NAMES_TO_OBJS = {}
REGISTERED_OBJS_TO_NAMES = {}


def register_internal_serializable(path, symbol):
    global REGISTERED_NAMES_TO_OBJS
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    REGISTERED_NAMES_TO_OBJS[name] = symbol
    REGISTERED_OBJS_TO_NAMES[symbol] = name

if namex:

    class twotower_export(namex.export):
        def __init__(self, path):
            super().__init__(package="twotower", path=path)

        def __call__(self, symbol):
            register_internal_serializable(self.path, symbol)
            return super().__call__(symbol)
        
else:

    class twotower_export:
        def __init__(self, path):
            self.path = path

        def __call__(self, symbol):
            register_internal_serializable(self.path, symbol)
            return symbol