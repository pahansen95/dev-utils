"""

OpenAI Provided Models

"""

from __future__ import annotations
from collections.abc import Generator, Callable
from typing import ContextManager, TypeVar, Generic
from functools import wraps, update_wrapper
import requests, contextlib, logging
from .core import *

logger = logging.getLogger(__name__)

CHAT_MODELS: dict[str, Model] = {}
EMBED_MODELS: dict[str, Model] = {}

@dataclass
class GPT(Model):
  name: Literal['gpt-4o', 'gpt-4o-mini']

CHAT_MODELS |= { m.name: m for m in (
  GPT(
    'chatgpt',
    {
      'version': 'chatgpt-4o-latest',
      'inputSize': 128_000,
      'outputSize': 16_384,
    }
  ),
  GPT(
    'gpt-4o',
    {
      'version': 'gpt-4o',
      'inputSize': 128_000,
      'outputSize': 16_384,
    }
  ),
  GPT(
    'gpt-4o-mini',
    {
      'version': 'gpt-4o-mini',
      'inputSize': 128_000,
      'outputSize': 16_384,
    }
  ),
) }

@dataclass
class Reasoning(Model):
  name: Literal['o1', 'o1-mini', 'o3-mini']

CHAT_MODELS |= { m.name: m for m in (
  Reasoning(
    'o1',
    {
      'version': 'o1',
      'inputSize': 200_000,
      'outputSize': 100_000,
    },
    {
      'reasoning_effort': 'low'
    },
  ),
  Reasoning(
    'o1-mini',
    {
      'version': 'o1-mini',
      'inputSize': 128_000,
      'outputSize': 65_536,
    },
    {
      'reasoning_effort': 'low'
    },
  ),
  Reasoning(
    'o3-mini',
    {
      'version': 'o3-mini',
      'inputSize': 200_000,
      'outputSize': 100_000,
    },
    {
      'reasoning_effort': 'low'
    },
  ),
) }

@dataclass
class TextEmbedding(Model):
  name: Literal['text-embedding-3-large', 'text-embedding-3-small']

EMBED_MODELS |= {
  'text-embedding-3-large': TextEmbedding(
    'text-embedding-3-large',
    {
      'version': 'text-embedding-3-large',
      'inputSize': 8192,
      'outputSize': 3072,
      'outputDType': 'f32',
    }
  ),
  'text-embedding-3-small': TextEmbedding(
    'text-embedding-3-small',
    {
      'version': 'text-embedding-3-small',
      'inputSize': 8192,
      'outputSize': 1536,
      'outputDType': 'f32',
    }
  ),
}

class MessageV1(TypedDict):
  role: Literal['system', 'user', 'assistant']
  content: str
  name: NotRequired[str]

class MessageV2(TypedDict):
  role: Literal['developer', 'user', 'assistant']
  content: str
  name: NotRequired[str]

MSG_T = type[MessageV1 | MessageV2]

REQ_RESP_T = tuple[
  tuple[int, int],
  requests.Response,
]

@dataclass
class OpenAISession(ProviderSession):
  url: str = 'https://api.openai.com/v1'
  session: requests.Session = field(default_factory=requests.session())
  headers: dict[str, str] = field(default_factory=dict)

  @contextlib.contextmanager
  def json_request(self,
    route: str,
    body: dict,
    headers: dict = {},
    method = 'POST',
  ) -> Generator[REQ_RESP_T, None, None]:
    url = self.url.rstrip('/')
    route = route.lstrip('/')
    with self.session.request(
      method, f'{url}/{route}',
      headers=( headers | self.headers ), # TODO: Inject Auth Headers
      json=body,
    ) as resp:
      yield (
        divmod(resp.status_code, 100),
        resp,
      )
  
  @Retry()
  def retry_json_request(self,
    route: str,
    body: dict,
    headers: dict = {},
    method = 'POST',
  ) -> RETRY_T | dict:
    with self.json_request(route, body, headers=headers, method=method) as (
      (major, minor),
      resp,
    ):
      if major in { 5 }: return Retry
      elif major in { 4 }: raise ProviderError( obj=( { 'kind': 'error' } | resp.json() ) )
      assert major in { 2 }
      return { 'kind': 'reply' } | resp.json()

@dataclass
class OpenAIProvider(ModelProvider):
  models: dict[str, GPT | Reasoning | TextEmbedding] = field(default_factory=lambda: CHAT_MODELS | EMBED_MODELS)
  session: OpenAISession = field(default_factory=OpenAISession)
  _: KW_ONLY

  def chat(self, model_name: str, *messages: dict) -> str:
    model = self.models[model_name]

    if isinstance(model, GPT):
      ... # TODO: Convert Library Messages to OpenAI V1 Messages
    elif isinstance(model, Reasoning):
      ... # TODO: Convert Library Messages to OpenAI V2 Messages
    else: raise TypeError(type(model))

    try:
      resp = self.session.retry_json_request(
        route='/chat/completions',
        body=( model.props | {
          'model': model.cfg['version'],
          'messages': messages,
        } ),
      )
      assert resp['kind'] == 'reply'
    except ProviderError as e:
      assert e.obj['kind'] == 'error'
      err_msg = f'OpenAI Provider Error: {e}'
      logger.debug(err_msg)
      raise RuntimeError(err_msg) from e
    else: return ['choices'][0]['message']['content']

  def embed(self):
    ...
