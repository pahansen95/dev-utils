
from __future__ import annotations
from typing import TypeVar, Generic, Any, TypedDict, NotRequired, Required
from types import *
from collections.abc import *

### SemanticType Meta
class SemanticType:
  def __init__(self): raise RuntimeError # Don't allow instantiation of SemanticTypes