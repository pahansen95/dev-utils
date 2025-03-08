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
from SemanticTyping import SemanticType, Semantic, Implements, ClsVar, Factory
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

## Python's Data Model

We'll briefly review Python's data (object) model as it relates to Semantic Typing.

Most everything in Python is an object; an object is primarily a container for holding attributes, associated methods & references for gc.

Our type system deals in Python classes, which are also objects. The procedural order of a participating class - a semantic type - generally follows:

- First the developer defines a Semantic Type
  - A Class Definition is written:
    ```Python
    class BoxValue[T](Semantic):
      val: T
    ```
    > NOTE: Semantic is a sentinel indicating the class is part of the developer's mental model & not concrete.
  - Upon parsing by the Python Interpreter, the a namespace (dictionary of names & their values) is created from the class definition:
    `ns = { 'val': <Generic[T]> }`
  - The Metaclass is then determined; in this example Python infers the metaclass from the `Semantic` base of `BoxValue` (which is `SemanticType`).    
    > NOTE: Metaclasses are also objects so they have already undergone the process which we are describing here
  - The class object is then created wherein the Metaclass hooks are called in `SemanticType`
    > NOTE: Type Creation
    > To contextualize in terms of regular Python, a class object is how Python represents a type; Python uses the metaclass as the factory for producing new types.
    > By default the metaclass is the builting `type` Type which is provided as part of Python.
    > In a custom typing system, like Semantic Typing, a custom type factory is provided; `SemanticType` in our case.
    - `SemanticType.__new__( <Class[SemanticType]>, 'BoxValue', ( `Semantic`, ), ns ) -> <Class[BoxValue]>` is called to create the class object.
    - `SemanticType.__init( <Class[BoxValue]>, 'BoxValue', ( `Semantic`, ), ns )` is called to further configure the created class object.
- Next the user implements a concrete version of BoxValue (again as an object):
  - Again a Class Definition is formed
    ```Python
    class IntBox(Implements, BoxValue[int]):
      val: int
    ```
    > NOTE: `Implements` is another sentinel indicating the class is part of the developer's computational model & concretely implements semantic types.
  - Like before, the inferred metaclass hooks are called to create the class object; both `BoxValue` & `Implements` have the metaclass of `SemanticType`
- Then the user instantiates an instance of the concrete class (again as an object):
  - The Class factory is called
    ```Python
    my_number = IntBox(10)
    ```
  - The `__new__` & `__init__` methods are called just like a normal Python object.
'''
