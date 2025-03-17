"""

Implements the Anthropic Model Provider

"""
from __future__ import annotations
from .core import *
from .Backend.http import *
from ..chat import ChatMessage
import requests

@dataclass
class Claude(ChatModel):
  """Anthropic's Claude Model"""
  name: Literal['claude-3.5-sonnet']
  """Identifying Name of the Model"""

@dataclass
class ClaudeThinking(Claude):
  """Anthropic's Claude Model w/ Thinking Capabilites"""
  name: Literal['claude-3.7-sonnet']
  """Identifying Name of the Model"""

CHAT_MODELS = { m.name: m for m in (
  Claude('claude-3.5-haiku',
    {
      'version': 'claude-3-5-haiku-20241022',
      'inputSize': 200_000,
      'outputSize': 8_192,
    },
  ),
  Claude('claude-3.5-sonnet',
    {
      'version': 'claude-3-5-sonnet-20241022',
      'inputSize': 200_000,
      'outputSize': 8_192,
    },
  ),
  Claude('claude-3.7-sonnet',
    {
      'version': 'claude-3-7-sonnet-20250219',
      'inputSize': 200_000,
      'outputSize': 8_192,
    },
  ),
  ClaudeThinking('claude-3.7-sonnet-think',
    {
      'version': 'claude-3-7-sonnet-20250219',
      'inputSize': 200_000,
      'outputSize': 64_000, # Includes Thinking Tokens
    },
    opts={
      # See Docs on extended thinking: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
      'thinking': {
        'budget_tokens': 16_000, # NOTE: 21_333 Tokens is the Maximum amount of thinking tokens before streaming is required.
        'type': 'enabled',
      }
    }
  ),
) }

# p.Message
class Message(TypedDict):
  """A ChatMessage in Anthropic's API"""
  role: Literal['user', 'assistant']
  content: str

  @staticmethod
  def transform(msg: c.ChatMessage) -> Message:
    _msg: Message = { 'content': msg['content'] }
    if msg['role'] in { c.Role.PLATFORM, c.Role.DEVELOPER }: raise ValueError('Anthropic Provider does not support system messages')
    elif msg['role'] in { c.Role.USER, }: _msg['role'] = 'user'
    elif msg['role'] in { c.Role.AGENT, c.Role.OTHER }: _msg['role'] = 'assistant'
    else: raise NotImplementedError(c.Role.name)
    return _msg

@dataclass
class AnthropicEndpoints:
  _: KW_ONLY
  chat: str = '/messages'

@dataclass
class AnthropicCfg(p.ProviderCfg):
  """Anthropic's Provider Config"""
  _: KW_ONLY
  url: str = field(default='https://api.anthropic.com/v1')
  api_version: Literal['2023-06-01'] = field(default='2023-06-01')
  endpoints: AnthropicEndpoints = field(default_factory=AnthropicEndpoints)

@dataclass(frozen=True)
class AnthropicAuth(p.ProviderAuth):
  """Anthropic's API Authentication"""
  token: str
  _: KW_ONLY
  key: str = field(default='x-api-key')

  @cache
  def to_http_headers(self) -> dict: return { self.key: self.token }

RESP_T = tuple[tuple[int, int], requests.Response]

@dataclass
class AnthropicSession(p.ProviderSession, HTTPBackend):
  """Anthropic API Session"""
  cfg: AnthropicCfg
  _ : KW_ONLY
  url: str = field(init=False)

  def __post_init__(self):
    self.url = self.cfg.url

  def _headers(self, **kwargs) -> dict[str, str]: return super()._headers(**{
    'anthropic-version': self.cfg.api_version, **kwargs
  })


@dataclass
class Anthropic(BaseModelProvider):
  """Anthropic Model Provider"""
  models: dict[str, Claude | ClaudeThinking]
  cfg: AnthropicCfg
  session: AnthropicSession
  
  def chat(self, model: str, *messages: c.ChatMessage) -> c.ChatMessage:
    _chat_endpoint = self.cfg.endpoints.chat
    _model = self.models[model]
    assert isinstance(_model, Claude), _model
    system_msgs = [
      {
        'type': 'text', 'text': msg['content'],
      } for msg in messages
      if msg['role'] in { c.Role.PLATFORM, c.Role.DEVELOPER }
    ]
    content_msgs = list(map(Message.transform, (
      msg for msg in messages
      if msg['role'] not in { c.Role.PLATFORM, c.Role.DEVELOPER }
    )))
    try:
      resp = self.session.retry_request(
        path=_chat_endpoint,
        body=( _model.opts | {
          'system': system_msgs,
          'messages': content_msgs,
          'model': _model.cfg['version'],
          'max_tokens': _model.cfg['outputSize'],
        } ),
      )
      assert resp['kind'] == 'response'
    except ProviderHTTPBackendError as e:
      assert e.obj['kind'] == 'error'
      err_msg = f'Anthropic Provider Error: {e}'
      logger.debug(err_msg)
      raise RuntimeError(err_msg) from e
    else:
      resp_content = resp['content']
      assert isinstance(resp_content, list) and len(resp_content) <= 2
      content = next(x['text'] for x in resp_content if x['type'] == 'text')
      reply = { 'role': c.Role.AGENT, 'content': content, 'props': {
        'author': f'{model}',
        'agent_requested_version': _model.cfg['version'],
        'agent_provider_version': resp.get('model', None),
        'created_at': time.time_ns(),
      } }
      if 'thinking' in _model.opts: reply['props']['thinking'] = [
        x for x in resp_content
        if x['type'] in {'thinking', 'redacted_thinking'}
      ]
      return reply

### Model Loader Interface

def load_provider_models(cfg: ModelCfg) -> dict[str, Claude | ClaudeThinking]:
  return CHAT_MODELS

def load_provider_config(
  env: dict[str, str] | None,
  partial: dict[str, str] | None,
) -> AnthropicCfg:
  if env is None and partial is None: raise ValueError("Must provide at least one of env or partial")
  if partial is None: partial = json.loads(load_env(*(
      'DEVAGENT_PROVIDER_ANTHROPIC_CFG',
      'ANTHROPIC_CFG',
    ), env=env, default='{}'))
  MISSING = type('MISSING', (), {})
  def _pop(k, d = MISSING, o: dict = partial): return o.pop(k, d)
  cfg = AnthropicCfg()
  if (url := _pop('url')) is not MISSING: cfg.url = url
  if (api_version := _pop('api_version')) is not MISSING: cfg.api_version = api_version
  if (endpoints := _pop('endpoints')) is not MISSING:
    assert isinstance(endpoints, dict)
    logger.debug(f'{endpoints=}')
    if (chat_endpoint := _pop('chat', o=endpoints)): cfg.endpoints.chat = chat_endpoint
  return cfg

def load_provider_auth(env: dict[str, str]) -> AnthropicAuth:
  """Loads the API Token from the passed env"""
  token = load_env(*(
    'DEVAGENT_PROVIDER_ANTHROPIC_TOKEN',
    'ANTHROPIC_TOKEN',
  ), env=env)
  return AnthropicAuth(token=token)

def load_provider_session(
  cfg: AnthropicCfg,
  auth: AnthropicAuth,
) -> AnthropicSession:
  return AnthropicSession(
    cfg=cfg,
    auth_headers=auth.to_http_headers,
  )

def load_provider(
  auth_env: Mapping[str, str],
  cfg_env: Mapping[str, str] = None,
  partial_cfg: Mapping[str, str] = None,
) -> Anthropic:
  if partial_cfg is None and cfg_env is None: raise ValueError('Must provide one of cfg_env or partial_cfg')
  cfg = load_provider_config(cfg_env, partial_cfg)
  models = load_provider_models(cfg)
  auth = load_provider_auth(auth_env)
  session = load_provider_session(cfg, auth)
  return Anthropic(
    models=models,
    cfg=cfg,
    session=session,
  )
