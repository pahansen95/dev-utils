from __future__ import annotations
from typing import TypedDict, NotRequired, Callable
from collections.abc import MutableMapping

import logging, copy

logger = logging.getLogger(__name__)

### Semantic Typing System

class SemanticTypeAttr(TypedDict):
  props: SemanticTypeProps

class SemanticTypeProps(TypedDict):
  concrete: bool

class SemanticType(type):
  """The Base Meta Type of the Semantic Typing System."""

  __semantic_typing__: SemanticTypeAttr = {
    'props': {
      'concrete': None,
    },
  }

  def __new__(mcls, name: str, bases: tuple[type, ...], namespace: MutableMapping, **kwargs):
    """Called once during the creation of any new class (ie. name) whose metaclass is this Type"""

    ### Create the Class
    cls = super().__new__(mcls, name, bases, namespace, **kwargs)

    ### Merge the Semantic Type Attributes
    
    mattr = copy.deepcopy(mcls.__semantic_typing__) 
    # First merge in the Semantic Types
    for base in filter(
      lambda b: issubclass(b, Semantic),
      bases,
    ):
      base_mattr: SemanticTypeAttr = getattr(base, '__semantic_typing__')
      assert isinstance(base_mattr, dict)
      for k, v in base_mattr.items(): mattr[k] |= v
    # Next Merge in the Computational Types
    for base in filter(
      lambda b: issubclass(b, Implements),
      bases,
    ):
      base_mattr: SemanticTypeAttr = getattr(base, '__semantic_typing__')
      assert isinstance(base_mattr, dict)
      for k, v in base_mattr.items(): mattr[k] |= v
    
    ### Run Usage Validations

    # Determine if Semantic or Computational
    if ( concrete := kwargs.get('concrete') ) is not None:
      if (
        not concrete # Class is Semantic
          and
        mattr['props']['concrete'] # But SubClass is declared as concrete
      ): raise TypeError("A Semantic Type can't extend a Computational Type")
      mattr['props']['concrete'] = concrete or mattr['props']['concrete']
    if not isinstance(mattr['props']['concrete'], bool):
      raise TypeError(f'Semantic Typing System expects a boolean type for props.concrete: got `{type(mattr['props']['concrete'])}`')
    
    ### Apply the Attributes

    setattr(cls, '__semantic_typing__', mattr)

    ### Return the new class

    logger.debug(f"Creating class {name}: Semantic Typing Attributes: {mattr}")
    return cls

class Semantic(metaclass=SemanticType, concrete=False):
  """A sentinel indicating the class is a semantic type"""
  # TODO: Implement protocol like functionality to enable structural typing

class Implements(metaclass=SemanticType, concrete=True):
  """A sentinel indicating a class is a computational type (implementing a semantic type)"""
  # TODO: Implement dataclass like functionality to autopopulate instance properties based on class attributes

### Dataclass Like Constructs
class ClsVar[T]:
  """A sentinel indicating the variable is a class variable instead of an instance variable."""
class Factory[T]:
  """A Factory for a variable's value; the factory function accepts the class type being instantiated"""
  def __init__(self, new: Callable[[type], T]): self.new = new
  def __call__(self) -> T: return self.new()

### Example Usage

from SemanticTyping import Semantic, SemanticType, Implements, ClsVar, Factory

class BoxValue[T](Semantic):
  val: T

class PersitentValue[T](BoxValue[T]):
  @staticmethod
  def marshal(v: T) -> bytes: ...
  @staticmethod 
  def unmarshal(b: bytes) -> T: ...

class ValueGetter[T](Semantic):
  def get(self) -> T: ...

class ConcreteNumberValue(Implements, PersitentValue[int], ValueGetter[int]):
  val: int
  def get(self) -> int: return self.val
  @staticmethod
  def marshal(v: int) -> bytes: return int.to_bytes(v, length=( 128 // 8 ), byteorder='big', signed=True)
  @staticmethod
  def unmarshal(b: bytes) -> int: return int.from_bytes(b, byteorder='big')

class ConcreteU16Value(ConcreteNumberValue):
  val: int = 10
  bits: ClsVar[int] = Factory(lambda cls: 16)
  @staticmethod
  def marshal(v: int) -> bytes: return int.to_bytes(v, length=( ConcreteU16Value.bits // 8 ), byteorder='big', signed=False)
