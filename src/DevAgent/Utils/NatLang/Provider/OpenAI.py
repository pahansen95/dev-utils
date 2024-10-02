"""

Implements a Model Provider for the Public OpenAI API

"""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, KW_ONLY, field
import urllib.parse, logging, json, functools, os, base64

### Pkg Imports
from ....Utils import get_chain
from .. import ModelProvider, ProviderCfg, ProviderError, ModelCfg
from ..chat import Message
###

logger = logging.getLogger(__name__)

def load_chat_config(**o: str) -> ModelCfg:
  return json.loads(o.get('DEVAGENT_PROVIDER_OPENAI_CHAT_CFG', json.dumps({
    'model': 'gpt-4o-2024-08-06',
    'inputSize': 128_000,
    'outputSize': 16_384
  }))) | {
    'endpoint': urllib.parse.urlparse(
      o.get('DEVAGENT_PROVIDER_OPENAI_CHAT_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
    ),
    'token': get_chain(o,
      'DEVAGENT_PROVIDER_OPENAI_CHAT_TOKEN',
      'DEVAGENT_PROVIDER_OPENAI_TOKEN'
    ),
  }

def load_embed_config(**o: str) -> ModelCfg:
  return json.loads(o.get('DEVAGENT_PROVIDER_OPENAI_EMBED_CFG', json.dumps({
    'model': 'text-embedding-3-large',
    'inputSize': 8192,
    'outputSize': 3072,
    'outputDType': 'float32',
  }))) | {
    'endpoint': urllib.parse.urlparse(
      o.get('DEVAGENT_PROVIDER_OPENAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
    ),
    'token': get_chain(o,
      'DEVAGENT_PROVIDER_OPENAI_EMBED_TOKEN',
      'DEVAGENT_PROVIDER_OPENAI_TOKEN',
    )
  }

def load_provider_config(**o: str) -> ProviderCfg:
  """Loads the OpenAI Provider Config"""
  return json.loads(o.get('DEVAGENT_PROVIDER_OPENAI_CFG', json.dumps({
    'name': 'OpenAI'
  })))

@dataclass
class OpenAIProvider(ModelProvider):
  """Model Provider using the OpenAI Public API"""
  cfg: ProviderCfg

  ### Embedding Interface ###

  def embed_req_headers(self) -> dict[str, str]:
    return self.cfg.get('httpHeaders', {}) | self.cfg['embed'].get('httpHeaders', {}) | {
      'Authorization': f'Bearer {self.cfg['embed']['token']}'
    }
  def embed_req_body(self, *inputs, **kwargs) -> dict[str, Any]:
    return self.cfg['embed'].get('props', {}) | {
      'input': list(inputs),
      'model': self.cfg['embed']['model'],
      'dimensions': self.cfg['embed']['outputSize'],
      'encoding_format': 'base64',
    }
  def embed_extract_content(self, body: bytes) -> list[bytes]:
    return list(map(base64.b64decode, (d['embedding'] for d in json.loads(body)["data"])))
  def embed_extract_error(self, body: bytes) -> ProviderError:
    return { 'kind': 'error' } | json.loads(body)

  ### Chat Interface ###

  def chat_req_headers(self) -> dict[str, str]:
    return self.cfg.get('httpHeaders', {}) | self.cfg['chat'].get('httpHeaders', {}) | {
      'Authorization': f'Bearer {self.cfg['chat']['token']}'
    }
  def chat_req_body(self, *messages: Message, **kwargs) -> dict[str, Any]:
    return self.cfg['chat'].get('props', {}) | {
      'model': self.cfg['chat']['model'],
      'messages': list(messages),
      'stream': False,
    }
  def chat_extract_content(self, body: bytes) -> list[bytes]:
    return json.loads(body)['choices'][0]['message']['content']
  def chat_extract_error(self, body: bytes) -> ProviderError:
    return { 'kind': 'error' } | json.loads(body)
