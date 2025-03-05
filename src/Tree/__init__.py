"""

Tree DataStructures

"""
import logging

### Specify what names to export
__all__ = [
  # Tree Essentials
  Node, Edge, Tree,
  # SubPkgs
  'diff', # Tree Diffs
]

logger = logging.getLogger(__name__)

### To avoid cyclical dependencies, add local imports below

from .tree import *
from . import diff
