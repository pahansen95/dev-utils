from typing import *
from collections.abc import *
from types import *

import logging
from dataclasses import dataclass, field, fields, KW_ONLY

logger = logging.getLogger(__name__)

STRUCTURAL = type('STRUCTURAL', (), {})
"""Sentinel to indicate some topological construct in a tree"""
WALK_T = Literal['dfs:pre']
"""Tree Traversal Algorithms"""