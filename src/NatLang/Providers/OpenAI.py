"""

OpenAI Provided Models

"""

from __future__ import annotations
from collections.abc import Generator, Callable
from typing import ContextManager, TypeVar, Generic
from functools import wraps, update_wrapper
import requests, contextlib, logging, base64
from .core import *
from .Backend.http import *

logger = logging.getLogger(__name__)

CHAT_MODELS: dict[str, p.Model] = {}
EMBED_MODELS: dict[str, p.Model] = {}

@dataclass
class GPT(ChatModel):
  name: Literal['gpt-4.5', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'chatgpt']
  _: KW_ONLY
  caps: ModelCapabilities = field(init=False, default_factory=lambda: model_capabilities(chat=True))

CHAT_MODELS |= { m.name: m for m in (
  GPT(
    'gpt-4.1',
    {
      'version': 'gpt-4.1-2025-04-14',
      'inputSize': 1_047_576,
      'outputSize': 32_768,
    }
  ),
  GPT(
    'gpt-4.1-mini',
    {
      'version': 'gpt-4.1-mini-2025-04-14',
      'inputSize': 1_047_576,
      'outputSize': 32_768,
    }
  ),
  GPT(
    'gpt-4.1-nano',
    {
      'version': 'gpt-4.1-nano-2025-04-14',
      'inputSize': 1_047_576,
      'outputSize': 32_768,
    }
  ),
  GPT(
    'gpt-4.5',
    {
      'version': 'gpt-4.5-preview-2025-02-27',
      'inputSize': 128_000,
      'outputSize': 16_384,
    }
  ),
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
class Reasoning(ChatModel):
  name: Literal['o1', 'o1-mini', 'o3-mini']
  _: KW_ONLY
  caps: ModelCapabilities = field(init=False, default_factory=lambda: model_capabilities(chat=True))

CHAT_MODELS |= { m.name: m for m in (
  Reasoning(
    'o1',
    {
      'version': 'o1',
      'inputSize': 200_000,
      'outputSize': 100_000,
    },
    opts={
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
    opts={
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
    opts={
      'reasoning_effort': 'low'
    },
  ),
) }

@dataclass
class TextEmbedding(EmbedModel):
  name: Literal['text-embedding-3-large', 'text-embedding-3-small']
  _: KW_ONLY
  caps: ModelCapabilities = field(init=False, default_factory=lambda: model_capabilities(embed=True))

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


class _Message(TypedDict):
  role: str
  content: str
  name: NotRequired[str]

  @staticmethod
  def transform(msg: c.ChatMessage) -> MSG_T:
    _msg: MSG_T = { 'content': msg['content'] }
    if 'author' in msg.get('props', {}): _msg['name'] = msg['props']['author']
    if msg['role'] in { c.Role.PLATFORM, c.Role.DEVELOPER }: _msg['role'] = '' # Passthrough
    elif msg['role'] in { c.Role.USER, }: _msg['role'] = 'user'
    elif msg['role'] in { c.Role.AGENT, c.Role.OTHER }: _msg['role'] = 'assistant'
    else: raise NotImplementedError(c.Role.name)
    return _msg

class MessageV1(_Message, TypedDict):
  role: Literal['system', 'user', 'assistant']
  @staticmethod
  def transform(msg: c.ChatMessage) -> MessageV1:
    _msg: MessageV1 = _Message.transform(msg)
    if msg['role'] in { c.Role.PLATFORM, c.Role.DEVELOPER }: _msg['role'] = 'system'
    return _msg

class MessageV2(_Message, TypedDict):
  role: Literal['developer', 'user', 'assistant']
  @staticmethod
  def transform(msg: c.ChatMessage) -> MessageV2:
    _msg: MessageV2 = _Message.transform(msg)
    if msg['role'] in { c.Role.PLATFORM, c.Role.DEVELOPER }: _msg['role'] = 'developer'
    return _msg

MSG_T = type[MessageV1 | MessageV2]

@dataclass
class OpenAIEndpoints:
  _: KW_ONLY
  chat: str = '/chat/completions'
  embed: str = '/embeddings'

@dataclass
class OpenAICfg(p.ProviderCfg):
  _: KW_ONLY
  url: str = 'https://api.openai.com/v1'
  endpoints: OpenAIEndpoints = field(default_factory=OpenAIEndpoints)

@dataclass(frozen=True)
class OpenAIAuth(p.ProviderAuth):
  token: str
  _: KW_ONLY
  organization: str | None = None
  project: str | None = None

  @cache
  def to_http_headers(self) -> dict[str, str]:
    auth_headers = { 'Authorization': f'Bearer {self.token}' }
    if self.organization is not None: auth_headers['OpenAI-Organization'] = self.organization
    if self.project is not None: auth_headers['OpenAI-Project'] = self.project
    return auth_headers

@dataclass
class OpenAISession(p.ProviderSession, HTTPBackend):
  """An API Session w/ the OpenAI HTTP based API"""
  _: KW_ONLY

@dataclass
class OpenAI(BaseModelProvider):
  models: dict[str, GPT | Reasoning | TextEmbedding]
  session: OpenAISession
  cfg: OpenAICfg

  def chat(self, model: str, *messages: c.ChatMessage) -> c.ChatMessage:
    _model = self.models[model]
    _chat_endpoint = self.cfg.endpoints.chat
    if isinstance(_model, GPT): _Message = MessageV1
    elif isinstance(_model, Reasoning): _Message = MessageV2
    else: raise TypeError(type(_model))
    _messages = list(map(_Message.transform, messages))
    try:
      resp = self.session.retry_request(
        path=_chat_endpoint,
        body=( _model.opts | {
          'model': _model.cfg['version'],
          'messages': _messages,
        } ),
      )
      assert resp['kind'] == 'response'
    except ProviderHTTPBackendError as e:
      assert e.obj['kind'] == 'error'
      err_msg = f'OpenAI Provider Error: {e}'
      logger.debug(err_msg)
      raise RuntimeError(err_msg) from e
    else: return { 'role': c.Role.AGENT, 'content': resp['choices'][0]['message']['content'], 'props': {
      'author': f'{model}',
      'agent_requested_version': _model.cfg['version'],
      'agent_provider_version': resp.get('model', None),
      'created_at': time.time_ns(),
    } }

  def embed(self, model: str, *content: p.CONTENT) -> p.Embedding:
    _model = self.models[model]
    _embed_endpoint = self.cfg.endpoints.embed
    assert isinstance(model, TextEmbedding)
    try:
      resp = self.session.retry_request(
        route=f"/{_embed_endpoint.lstrip('/')}",
        body=( _model.opts | {
          'model': _model.cfg['version'],
          'input': content,
          'dimensions': _model.cfg['outputSize'],
          'encoding_format': 'base64',
        } ),
      )
      assert resp['kind'] == 'response'
    except ProviderError as e:
      assert e.obj['kind'] == 'error'
      err_msg = f'OpenAI Provider Error: {e}'
      logger.debug(err_msg)
      raise RuntimeError(err_msg) from e
    else: return {
      'buffer': list(map(base64.b64decode, (d['embedding'] for d in resp["data"]))),
      # TODO...
    }

  @contextlib.contextmanager
  def tune(self, model: str, **opts) -> Generator:
    """Applies tuning options to the model"""
    old_opts = self.models[model].opts
    try:
      self.models[model].opts |= opts
      yield
    finally:
      self.models[model].opts = old_opts
    
  def supports(self, model: str, capability: str) -> bool: return self.models[model].caps[capability]

### Model Loader Interface

def load_provider_models(cfg: ChatModel) -> dict[str, GPT | Reasoning | TextEmbedding]:
  return CHAT_MODELS | EMBED_MODELS

def load_provider_config(
  env: dict[str, str] | None,
  partial: dict[str, str] | None,
) -> OpenAICfg:
  if env is None and partial is None: raise ValueError("Must provide at least one of env or partial")
  if partial is None: partial = json.loads(load_env(*(
      'DEVAGENT_PROVIDER_OPENAI_CFG',
      'OPENAI_CFG',
    ), env=env, default='{}'))
  MISSING = type('MISSING', (), {})
  def _pop(k, d = MISSING, o: dict = partial): return o.pop(k, d)
  cfg = OpenAICfg()
  if (url := _pop('url')) is not MISSING: cfg.url = url
  if (endpoints := _pop('endpoints')) is not MISSING:
    assert isinstance(endpoints, dict)
    logger.debug(f'{endpoints=}')
    if (chat_endpoint := _pop('chat', o=endpoints)): cfg.endpoints.chat = chat_endpoint
    if (embed_endpoint := _pop('embed', o=endpoints)): cfg.endpoints.embed = embed_endpoint
  return cfg

def load_provider_auth(env: dict[str, str]) -> OpenAIAuth:
  """Loads the API Token from the passed env"""
  token = load_env(*(
    'DEVAGENT_PROVIDER_OPENAI_TOKEN',
    'OPENAI_TOKEN',
  ), env=env)
  org = load_env(*(
    'DEVAGENT_PROVIDER_OPENAI_ORG',
    'OPENAI_ORG',
  ), env=env, default=None)
  project = load_env(*(
    'DEVAGENT_PROVIDER_OPENAI_PROJECT',
    'OPENAI_PROJECT',
  ), env=env, default=None)
  return OpenAIAuth(token=token, organization=org, project=project)

def load_provider_session(
  cfg: OpenAICfg,
  auth: OpenAIAuth,
) -> OpenAISession:
  return OpenAISession(
    url=cfg.url,
    auth_headers=auth.to_http_headers,
  )

def load_provider(
  auth_env: Mapping[str, str],
  cfg_env: Mapping[str, str] = None,
  partial_cfg: Mapping[str, str] = None,
) -> OpenAI:
  if partial_cfg is None and cfg_env is None: raise ValueError('Must provide one of cfg_env or partial_cfg')
  cfg = load_provider_config(cfg_env, partial_cfg)
  models = load_provider_models(cfg)
  auth = load_provider_auth(auth_env)
  session = load_provider_session(cfg, auth)
  return OpenAI(
    models=models,
    session=session,
    cfg=cfg,
  )
