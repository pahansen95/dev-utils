"""

Implements Functionality for the parsing, manipulation & output of natural language.

This package relies heavily on external APIs for Large Language Models.

"""

from .Protocols import extern as p # External Protocols
from .Providers.loader import * # Lazy Load Providers
from . import chat
from . import embed # Concrete Implementations