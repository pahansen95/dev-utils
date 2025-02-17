"""

Protocols are Semantic Types describing sets of state, functionality and ownership. A Semantic Type System is used to reason
about the real world through articulation of mental models. 

Concrete implementations which implement the set of state & functionality of a semantic type ARE of that semantic type

"""
from __future__ import annotations
from typing import TypeVar, Generic
from types import *
from collections.abc import *

### SemanticType Meta
class SemanticType:
  def __init__(self): raise RuntimeError # Don't allow instantiation of SemanticTypes

### Typing

T = TypeVar('T')
"""Any Type"""
K = TypeVar('K')
"""Key Type"""
ROLE = TypeVar('ROLE', bound=str)
"""An identity classification"""
CONTENT = TypeVar('CONTENT', bound=str)
"""Textual representation of semantics; ie. Natural Language"""
LATENT = TypeVar('LATENT', bound=ByteString|Sequence[int])
"""Numerical representation of semantics; ie. An Embedding"""

### Protocols describing state & ownership

class Node(SemanticType, Generic[T]):
  """A Node in a Graph. Represents an entity with a unique identity."""
  id: Hashable
  def __eq__(self, other: object) -> bool: ...
  def __hash__(self) -> int: ...
class Edge(SemanticType, Generic[T]):
  """A Directed Edge in a Graph. Represents a relationship between two Nodes."""
  u: Node[T]
  v: Node[T]
  k: K
  def __eq__(self, other: object) -> bool: ...
  def __hash__(self) -> int: ...
class Graph(SemanticType, Generic[T]):
  """A Graph represents a set of related entities connected by directed edges."""
  node: Set[Node[T]]
  edges: Set[Edge[T]] 
  def adjacency(self) -> Mapping[Node[T], Iterator[Node[T]]]: ...

class Properties(MutableMapping[str, T]):
  """Contextual Metadata"""

class Agent(SemanticType):
  """Some entity participating in a Conversation; for example a LLM or a user"""
  name: str
  props: Properties | None

class Message(SemanticType):
  """A Single Message in a Conversation"""
  role: ROLE
  content: CONTENT
  props: Properties | None

class Embedding(SemanticType):
  """A Batch of Latents; design based on a NumPy ndarray"""
  buffer: Sequence[LATENT]
  """The batch of Latents; usually a continuous buffer of memory"""
  shape: tuple[int, ...]
  """The shape of the batch of latents; expected to be (batch_size, *latent_dimensions)"""
  dtype: str
  """The Data Type of the buffer; ex. f32 or u8"""
  props: Properties | None

class Conversation(SemanticType, Graph[Message]):
  """A set of messages, forming causal relationships between each other"""
  def chat_logs(self) -> Iterator[ChatLog]: ...

class ChatLog(SemanticType, Sequence[Message]):
  """A causally ordered sequence of messages in a conversation; ie. A single path in the conversational graph"""

### Protocols describing functionality

class Chat(SemanticType):
  """A Prompt-Reply communication process"""
  def __call__(self, msg: Message) -> Message: ...

class Embed(SemanticType):
  """A Process to encode semantics from textual to numerical formats"""
  def __call__(self, chunk: CONTENT) -> Embedding: ...

__all__ = [

]