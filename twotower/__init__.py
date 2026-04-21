import os

__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

from twotower.api import *  # noqa: F403, E402
from twotower.api import __version__  # noqa: E402


del os