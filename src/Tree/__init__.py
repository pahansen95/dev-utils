"""

Tree DataStructures

"""
import logging

### Specify what names to export
__all__ = [
  # Core
  'STRUCTURAL', 'WALK_T',
  # Tree Essentials
  'TreeNode', 'TreeEdge', 'OrderedMultiTree',
  # SubPkgs
  'diff', # Tree Diffs
]

logger = logging.getLogger(__name__)

### To avoid cyclical dependencies, add local imports below
from .core import *
from .tree import *
from . import diff
