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

@dataclass()
class GPT(Model):
  name: Literal['gpt-4o', 'gpt-4o-mini']
  cfg: ModelCfg

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
  props: dict | None = None

CHAT_MODELS |= { m.name: m for m in (
  Reasoning(
    'o1',
    {
      'version': 'o1',
      'inputSize': 200_000,
      'outputSize': 100_000,
    }
  ),
  Reasoning(
    'o1-mini',
    {
      'version': 'o1-mini',
      'inputSize': 128_000,
      'outputSize': 65_536,
    }
  ),
  Reasoning(
    'o3-mini',
    {
      'version': 'o3-mini',
      'inputSize': 200_000,
      'outputSize': 100_000,
    }
  ),
) }

@dataclass
class TextEmbedding(Model):
  name: Literal['text-embedding-3-large', 'text-embedding-3-small']
  props: dict | None = None

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
      headers=( headers | self.headers ),
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
      elif major in { 4 }: resp.raise_for_status()
      assert major in { 2 }
      return resp.json()

@dataclass
class OpenAIProvider(ModelProvider):
  models: dict[str, GPT | Reasoning | TextEmbedding] = field(default_factory=lambda: CHAT_MODELS | EMBED_MODELS)
  session: OpenAISession = field(default_factory=OpenAISession)
  _: KW_ONLY

  def chat(self):
    resp = self.session.retry_json_request(
      '/chat/completions',
      {
        # TODO...
      },
    )
    # TODO: Extract what we need from the Content from the API Response

  def embed(self):
    ...
