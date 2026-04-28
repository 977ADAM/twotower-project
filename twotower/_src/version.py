from twotower.src.api_export import twotower_export

__version__ = "3.11.9"

@twotower_export("twotower.version")
def version():
    return __version__