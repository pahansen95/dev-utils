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

class Agent(Protocol):
  """Some entity participating in a Conversation; for example a LLM or a user"""
  name: str
  props: MutableMapping[str, Any] | None

class Message(Protocol):
  """A Single Message in a Conversation"""
  role: ROLE
  content: CONTENT
  props: MutableMapping[str, Any] | None

class Embedding(Protocol):
  """A Batch of Latents; design based on a NumPy ndarray"""
  buffer: Sequence[LATENT]
  """The batch of Latents; usually a continuous buffer of memory"""
  shape: Sequence[int]
  """The shape of the batch of latents; expected to be (batch_size, *latent_dimensions)"""
  dtype: str
  """The Data Type of the buffer; ex. f32 or u8"""
  props: MutableMapping[str, Any] | None

class Conversation(Protocol):
  """A set of messages, forming causal relationships between each other"""
  messages: set[Message]
  """The total set of messages in the Conversation"""

  def new_chat(self, msg: Message):
    """Start a new chat log rooted at the passed message"""
  def add_reply(self, msg: Message, to: Message):
    """Add the message as a reply to the specified message"""
  def chat_logs(self) -> Iterator[ChatLog]:
    """A lazy loaded sequence of Chat Logs available in the conversation"""

class ChatLog(Sequence[Message]):
  """A causally ordered sequence of messages in a conversation; ie. A single path in the conversational graph"""

### Protocols describing functionality

class SeDer(Protocol, Generic[T]):
  class Marshal(Protocol):
    def __call__(self, o: T) -> ByteString: ...
  class Unmarshal(Protocol):
    def __call__(self, b: ByteString) -> T: ...

class Chat(Protocol):
  """A Prompt-Reply turn contextualized to a chat log"""
  def __call__(self, model: str, *messages: Message) -> Message: ...

class Embed(Protocol):
  """A Process to encode semantics from textual to numerical formats"""
  def __call__(self, model: str, chunk: CONTENT) -> Embedding: ...

class ModelCfg(Protocol):
  version: str
  """The fully qualified versioned name identifying this model in the Provider API"""
  inputSize: int
  """The total allowed token input"""
  inputDType: str | None
  """Expected datatype of the input"""
  outputSize: int
  """The maximum allowed token output"""
  outputDType: str | None
  """Expected datatype of the output"""

class ModelCapabilities(Protocol):
  """Capabilties the Model Provides"""
  chat: bool
  embed: bool

class ModelOpts(Protocol):
  """Runtime Tunable Options of the Model"""

class Model(Protocol):
  """A Large Language Model"""
  name: str
  """Identifying Name of the Model"""
  cfg: ModelCfg
  """Static Configuration detailing Model"""
  opts: ModelOpts | None
  caps: ModelCapabilities

AvailableModels = Mapping[str, Model]

class ModelTuner(Protocol):
  """A Context manager to temporarily tune the runtime parameters of a Model"""
  model: Model
  props: MutableMapping[str, Any] | None

  def __enter__(self) -> Model: ...
  def __exit__(self, exc_type: type[Exception], exc_value: Exception, traceback: TracebackType): ...

class ProviderAuth(Protocol):
  """Authentication for a Provider Session"""

class ProviderSession(Protocol):
  """An stateful session w/ a Model Provider"""

class ProviderCfg(Protocol):
  """Provider Platform Configurations"""

class ModelProvider(Protocol):
  """A Provider of a Large Language Model"""
  models: dict[str, Model]
  cfg: ProviderCfg
  session: ProviderSession

  ### Method Protocols
  chat: Chat | None
  embed: Embed | None

  ### Embedded Methods
  def tune(self, model: str, **opts: Unpack[ModelOpts]) -> ContextManager[None]:
    """Applies tuning options to the model"""
  def supports(self, model: str, capability: str) -> bool:
    """Check if the model supports the provided capability"""

class ProviderLoader(Protocol):
  def load_provider(self, auth_env: Mapping[str, str], cfg_env: Mapping[str, str] | None, partial_cfg: Mapping[str, str] | None) -> ModelProvider: ...
  def load_provider_models(self, cfg: ModelCfg) -> AvailableModels: ...
  def load_provider_config(self, env: Mapping[str, str] | None, partial: Mapping[str, str] | None) -> ProviderCfg: ...
  def load_provider_auth(self, env: Mapping[str, str]) -> ProviderAuth: ...
  def load_provider_session(self, cfg: ProviderCfg, auth: ProviderAuth) -> ProviderSession: ...

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
  "ModelTuner",
  "Node",
  "MutableMapping[str, Any]",
  "ProviderLoader",
  "ProviderSession",
  "ROLE",
  "SeDer",
  "T",
]