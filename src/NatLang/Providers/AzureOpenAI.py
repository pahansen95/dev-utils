"""

Azure OpenAI Provided Models

"""

from __future__ import annotations
from .OpenAI import *

logger = logging.getLogger(__name__)

_AZUREOAI_CHAT_MODELS = CHAT_MODELS
# Filter out unused models
CHAT_MODELS = { k: v for k, v in _AZUREOAI_CHAT_MODELS.items() if k not in {
  'chatgpt',
} }
_AZUREOAI_EMBED_MODELS = EMBED_MODELS
EMBED_MODELS = { k: v for k, v in _AZUREOAI_EMBED_MODELS.items() if k not in {
  '', # TODO:
} }

@dataclass
class AzureOAIEndpoints:
  _: KW_ONLY
  chat: str = '/chat/completions'
  embed: str = '/embeddings'
  # For latest API Specs see: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs
  inference_api_version: str = '2025-02-01-preview'

@dataclass
class AzureOAICfg(p.ProviderCfg):
  url: str
  """The URL Endpoint to use"""
  _: KW_ONLY
  endpoints: AzureOAIEndpoints = field(default_factory=AzureOAIEndpoints)


@dataclass
class AzureOAIAuth(p.ProviderAuth):
  token: str

  def __hash__(self): return hash((self.token,))

  @cache
  def to_http_headers(self) -> dict[str, str]:
    return { 'api-key': self.token }

@dataclass
class AzureOAISession(OpenAISession): ...

@dataclass
class AzureOAI(OpenAI):
  models: dict[str, GPT | Reasoning | TextEmbedding]
  session: AzureOAISession
  cfg: AzureOAICfg

  def chat(self, model: str, *messages: c.ChatMessage) -> c.ChatMessage:
    _model = self.models[model]
    _chat_endpoint = f'/openai/deployments/{_model.name}/{self.cfg.endpoints.chat.lstrip('/')}'
    _api_version = self.cfg.endpoints.inference_api_version
    if isinstance(_model, GPT): _Message = MessageV1
    elif isinstance(_model, Reasoning): _Message = MessageV2
    else: raise TypeError(type(_model))
    _messages = list(map(_Message.transform, messages))
    try:
      resp = self.session.retry_request(
        path=_chat_endpoint,
        body=( _model.opts | {
          'messages': _messages,
        } ),
        params={
          'api-version': _api_version
        },
      )
      logger.debug(f'POST {_chat_endpoint}?api-version={_api_version}\n{json.dumps(resp, indent=2)}')
      assert resp['kind'] == 'response'
    except ProviderError as e:
      assert e.obj['kind'] == 'error'
      err_msg = f'AzureOAI Provider Error: {e}'
      logger.debug(err_msg)
      raise RuntimeError(err_msg) from e
    else: return { 'role': c.Role.AGENT, 'content': resp['choices'][0]['message']['content'], 'props': {
      'author': f'{model}',
      'agent_requested_version': _model.cfg['version'],
      'agent_provider_version': resp.get('model', None),
      'created_at': time.time_ns(),
    } }

  def embed(self, model: str, *content: p.CONTENT) -> p.Embedding:
    raise NotImplementedError

def load_provider_auth(env: dict[str, str]) -> AzureOAIAuth:
  """Loads the API Token from the passed env"""
  token = load_env(*(
    'DEVAGENT_PROVIDER_AZUREOAI_TOKEN',
    'AZUREOAI_TOKEN',
  ), env=env)
  return AzureOAIAuth(token=token)

def load_provider_config(
  env: dict[str, str] | None,
  partial: dict[str, str] | None,
) -> AzureOAICfg:
  if env is None and partial is None: raise ValueError("Must provide at least one of env or partial")
  if partial is None: partial = json.loads(load_env(*(
      'DEVAGENT_PROVIDER_AZUREOAI_CFG',
      'AZUREOAI_CFG',
    ), env=env, default='{}'))
  logger.debug(f'{partial=}')
  MISSING = type('MISSING', (), {})
  def _pop(k, d = MISSING, o: dict = partial): return o.pop(k, d)
  if (url := _pop('url')) is MISSING: raise ProviderError('AzureOpenAI Config requires the URL be configured')
  cfg = AzureOAICfg(url)
  if (endpoints := _pop('endpoints')) is not MISSING:
    assert isinstance(endpoints, dict)
    logger.debug(f'{endpoints=}')
    if (chat_endpoint := _pop('chat', o=endpoints)): cfg.endpoints.chat = chat_endpoint
    if (embed_endpoint := _pop('embed', o=endpoints)): cfg.endpoints.embed = embed_endpoint
  return cfg

def load_provider_models(cfg: ChatModel) -> dict[str, GPT | Reasoning | TextEmbedding]:
  return CHAT_MODELS | EMBED_MODELS

def load_provider_session(
  cfg: AzureOAICfg,
  auth: AzureOAIAuth,
) -> AzureOAISession:
  return AzureOAISession(
    url=cfg.url,
    auth_headers=auth.to_http_headers,
  )

def load_provider(
  auth_env: Mapping[str, str],
  cfg_env: Mapping[str, str] = None,
  partial_cfg: Mapping[str, str] = None,
) -> AzureOAI:
  if partial_cfg is None and cfg_env is None: raise ValueError('Must provide one of cfg_env or partial_cfg')
  cfg = load_provider_config(cfg_env, partial_cfg)
  models = load_provider_models(cfg)
  auth = load_provider_auth(auth_env)
  session = load_provider_session(cfg, auth)
  return AzureOAI(
    models=models,
    session=session,
    cfg=cfg,
  )