from __future__ import annotations
from types import *
from typing import *
from collections.abc import *
from dataclasses import dataclass, field, KW_ONLY
import enum, hashlib, json
from Utils.Tree import OrderedMultiTree, TreeEdge, TreeNode, STRUCTURAL

from .Protocols import intern as p

class ModelCapabilities(TypedDict):
  chat: bool
  embed: bool

def model_capabilities(**kwargs: Unpack[ModelCapabilities]) -> ModelCapabilities: return {
  'chat': False,
  'embed': False,
} | kwargs
