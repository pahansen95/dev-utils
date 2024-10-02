"""

Provides Functionality around Natural Language Manipulation like embedding & "chatting".

"""
from __future__ import annotations
from typing import TypedDict, Optional, Any, Literal
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
import requests, logging, urllib.parse

### Package Imports
from ..http import HTTPStatus
###

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json; charset=utf-8',
  'Connection': 'keep-alive',
  'Keep-Alive': 'timeout=900', # Keep alive for at least 15m
}

class UnsupportedProtocol(NotImplementedError):
  """Some Protocol interface is not Provided by the implementation"""

### TODO: Implement Provider Error Datastructure & Interface ###
class ProviderError(TypedDict, total=False):
  kind: Literal['error']

class ModelCfg(TypedDict, total=False):
  model: str
  """The Unique Identifier for the Model"""
  endpoint: urllib.parse.ParseResult
  inputSize: int
  outputSize: int
  token: str
  """The Authorization Token"""
  outputDType: Optional[str]
  """The DataType (ie. float64, float32, ...) of the Model Output; only needed for `Embedding Models`"""
  httpHeaders: Optional[dict[str, str]]
  """Model Specific Provider Headers & Header Overrides"""
  opts: Optional[dict[str, Any]]
  """Options to passthrough as part of the underlying API Request Body"""

class ProviderCfg(TypedDict, total=False):
  name: str
  """Some unique Identifier for the Provider; usually a Name"""
  chat: Optional[ModelCfg]
  """The Chat Model Config"""
  embed: Optional[ModelCfg]
  """The Embedding Model Config"""
  httpHeaders: Optional[dict[str, str]]
  """Model Agnostic Provider Headers & Header Overrides"""

class ModelProvider(ABC):
  """The Expected Protocol of the Backend Model"""
  cfg: ProviderCfg

  @property
  def chat_model_identifier(self) -> str:
    """Returns some unique identifier for the Chat Model"""
    return f'{self.cfg['name']}:{self.cfg['chat']['model']}'

  @property
  def embed_model_identifier(self) -> str:
    """Returns some unique identifier for the Chat Model"""
    return f'{self.cfg['name']}:{self.cfg['embed']['model']}'

  ### Embedding Protocol ###
  def embed_req_headers(self) -> dict[str, str]:
    """Embedding Interface: Construct the Request Headers"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Embedding Protocol')
  def embed_req_body(self, *inputs: str, **kwargs) -> dict[str, Any]:
    """Embedding Interface: Construct a Request Body"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Embedding Protocol')
  def embed_extract_content(self, body: bytes) -> list[bytes]:
    """Embedding Interface: Extract the Content from the Response"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Embedding Protocol')
  def embed_extract_error(self, body: bytes) -> ProviderError:
    """Embedding Interface: Extracts the Error from the Response"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Embedding Protocol')

  ### Chat Protocol ###
  def chat_req_headers(self) -> dict[str, str]:
    """Chat Interface: Construct the Request Headers"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Chat Protocol')
  def chat_req_body(self, *messages: chat.Message, **kwargs) -> dict[str, Any]:
    """Chat Interface: Construct a Request Body"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Chat Protocol')
  def chat_extract_content(self, body: bytes) -> list[bytes]:
    """Chat Interface: Extract the Content from the Response"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Chat Protocol')
  def chat_extract_error(self, body: bytes) -> ProviderError:
    """Chat Interface: Extracts the Error from the Response"""
    raise UnsupportedProtocol(f'{self.fully_qualified_name} does not support Chat Protocol')

class _Interface(ABC):
  provider: ModelProvider
  conn: requests.Session | None

  @contextmanager
  def _request(self, *args, **kwargs) -> Generator[tuple[HTTPStatus, requests.Response], None, None]:
    if self.conn is not None: resp_ctx = self.conn.request(*args, **kwargs)
    else: resp_ctx = requests.request(*args, **kwargs)
    with resp_ctx as resp:
      yield (HTTPStatus.factory(resp.status_code), resp)

### Package Imports to avoid Cyclic Imports
from . import chat, embed, Provider
###

def load_chat_interface(provider_name: str | None = None, **o) -> chat.ChatInterface:
  if provider_name is None: provider_name = Provider.get_provider_name_from_map('chat', **o)
  return chat.ChatInterface(
    provider=Provider.load_provider_by_name_from_map(provider_name, { 'chat', }, **o),
    conn=requests.Session(),
  )
def load_embed_interface(provider_name: str | None = None, **o) -> embed.EmbeddingInterface:
  if provider_name is None: provider_name = Provider.get_provider_name_from_map('embed',**o)
  return embed.EmbeddingInterface(
    provider=Provider.load_provider_by_name_from_map(provider_name, { 'embed', }, **o),
    conn=requests.Session(),
  )
# def load_all_interfaces(provider_name: str | None = None, **o) -> tuple[chat.ChatInterface, embed.EmbeddingInterface]:
#   if provider_name is None: provider_name = Provider.get_provider_name_from_map(**o)
#   provider: ModelProvider = Provider.load_provider_by_name_from_map(provider_name, { 'chat', 'embed' }, **o)
#   session = requests.Session()
#   return (
#     chat.ChatInterface(
#       provider=provider,
#       conn=session
#     ),
#     embed.EmbeddingInterface(
#       provider=provider,
#       conn=session
#     )
#   )
