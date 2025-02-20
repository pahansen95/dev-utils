'''\
# Semantic Typing

Semantic Typing is a structural typing system which articulates semantics:

- Functionality (ie. procedures or functions)
- State (ie. properties or data)
- Ownership (ie. rules of state posession & mutability)

## Summary

The term `semantics` precisely refers to a developer's mental models of the
systems they are modeling computationally. Semantic typing does not constrain
said computational model implemented, but it does elucidate & formalize the
system models of the domains implicit in the mental models.

> A `Computational Model` is a representation of a semantic system model that can
> be evaluated, executed and/or computed by a computer. A Computation Model is
> generally represented by a General Purpose Programming Language like Python.

It can help to consider Semantic Types as being an extension to Python Protocols (strutural typing metaclass)
providing meta-programming capabilities to develop a pDSL (Psuedo Domain Specific Language)
while still allowing for adaptation, extension or external intergrations of the pDSL
via Python's general purpose nature.

Semantic Typing should be used to paint mental boundaries & largely guide or direct pragmatic
development efforts. It should be used AFTER a developer has internalized a problem, model or 
program & wants to refactor or share their work; at this point the developer should have formed
their mental models. Semantic Typing becomes a tool to extract latent knowledge & realize it
in a format that can be analyzed & reasoned on in a group setting.

## Quick Start

```Python
from SemanticTyping import Semantic, SemanticMeta, Implements, ClsVar, Factory
from typing import *
from collections.abc import *

### First define the Mental Models (ie. Semantic Models)

class ConsumableResource[T](Semantic):
  resource_registry: set[T]
  def __enter__(self) -> T: ...
  def __exit__(self, exc_t, exc, tb): ...

class Server[T](Semantic):
  max_conn: int
  def bind(self, fd: int, addr: str): ...
  def listen(self, fd: int) -> T: ...
  def shutdown(self, fd: int): ...

### Next define the Computational Models
# NOTE: We've ommitted the concrete implementation for brevity

class UnixSocket(Implements, ConsumableResource[int], Server[BinaryIO]):
  resource_registry: set[int] = Factory(lambda _: set)
  """The file descriptors for all currently allocated Unix Sockets managed by the object"""
  max_conn: int = 10
  """The total number of active client connections per socket"""
  def __enter__(self) -> int:
    """Allocate a new Unix Socket & return the referrant file descriptor"""
    ...
  def __exit__(self, exc_t, exc, tb):
    """Close the Unix Socket & Cleanup all OS Resources"""
    ...
  def bind(self, fd: int, addr: str):
    """Binds the Unix Socket referred to by fd to the supplied address"""
    ...
  def listen(self, fd: int) -> BinaryIO:
    """Blocks listening for a connecting client & returns an IO object to communicate with them on connection"""
    ...
  def shutdown(self, fd: int):
    """Shutdown the server closing any open client connections & refusing any new connections; remains listening"""
    ...
```
'''
from __future__ import annotations
from typing import Protocol, TypedDict, NotRequired, Callable
from collections.abc import MutableMapping
import dataclasses

import logging, copy

logger = logging.getLogger(__name__)

### Semantic Typing System

class SemanticTypeAttr(TypedDict):
  props: SemanticTypeProps

class SemanticTypeProps(TypedDict):
  concrete: bool

class SemanticMeta(type):
  """The Base Meta Type of the Semantic Typing System."""

  __semantic_typing__: SemanticTypeAttr = {
    'props': {
      'concrete': None,
    },
  }

  def __new__(mcls, name: str, bases: tuple[type, ...], namespace: MutableMapping, **kwargs):

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
    
    ### Validations

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

    ### Return the class
    logger.debug(f"Creating class {name}: Semantic Typing Attributes: {mattr}")
    return cls

class Semantic(metaclass=SemanticMeta, concrete=False):
  """A sentinel indicating the class is a semantic type"""
  # TODO: Implement protocol like functionality to enable structural typing

class Implements(metaclass=SemanticMeta, concrete=True):
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

from SemanticTyping import Semantic, SemanticMeta, Implements, ClsVar, Factory

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
  def marshal(v: int) -> bytes: return int.to_bytes(v, length=( ConcreteNumberValue.bits // 8 ), byteorder='big', signed=False)
