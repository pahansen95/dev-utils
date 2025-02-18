"""

Externally Available Protocls

"""
from __future__ import annotations
from .core import *

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

  def add_nodes(self, *node: Node[T]): ...
  def add_edges(self, *edge: Edge[T]): ...
  def get_roots(self) -> Sequence[Node[T]]: ...

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

class SeDer(SemanticType, Generic[T]):
  class Marshal(SemanticType):
    def __call__(self, o: T) -> ByteString: ...
  class Unmarshal(SemanticType):
    def __call__(self, b: ByteString) -> T: ...

class Chat(SemanticType):
  """A Prompt-Reply turn contextualized within a chat"""
  def __call__(self, *msg: Message) -> Message: ...

class Embed(SemanticType):
  """A Process to encode semantics from textual to numerical formats"""
  def __call__(self, chunk: CONTENT) -> Embedding: ...

class Model(SemanticType):
  """A Large Language Model"""
  name: str
  """Identifying Name of the Model"""
  props: Properties | None
  """Default Runtime Properties of the Model"""

  ### Method Protocols
  chat: Chat | None
  embed: Embed | None

  ### Embedded Methods
  def parse_instructions(self, *instruct: Message) -> Sequence[Message]:
    """Converts the set of model instructions into a format expected by the model"""

class ModelTuner(SemanticType):
  """A Context manager to temporarily tune the runtime parameters of a Model"""
  model: Model
  props: Properties | None

  def __enter__(self) -> Model: ...
  def __exit__(self, exc_type: type[Exception], exc_value: Exception, traceback: TracebackType): ...

class ModelProvider(SemanticType):
  """A Provider of a Large Language Model"""
  models: dict[str, Model]

class ModelSession(SemanticType):
  """An Stateful session w/ a Model via it's provider"""
  name: str
  """The Model Name"""
  provider: ModelProvider
  """The Model Provider"""


__all__ = [
  "Agent",
  "CONTENT",
  "Chat",
  "ChatLog",
  "Conversation",
  "Edge",
  "Embed",
  "Embedding",
  "Graph",
  "K",
  "LATENT",
  "Message",
  "Model",
  "ModelProvider",
  "ModelSession",
  "ModelTuner",
  "Node",
  "Properties",
  "ROLE",
  "SeDer",
  "T",
]