"""

Implements a Model Provider for the Public Anthropic API

NOTE:

- Anthropic does not provide an Embedding Model

"""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, KW_ONLY, field
import urllib.parse, logging, json, functools, os, base64

### Pkg Imports
from ....Utils import get_chain
from .. import ModelProvider, ProviderCfg, ProviderError, ModelCfg
from ..chat import Message, SYSTEM_PROMPT
###

logger = logging.getLogger(__name__)

def load_chat_config(**o: str) -> ModelCfg:
  return json.loads(o.get('DEVAGENT_PROVIDER_ANTHROPIC_CHAT_CFG', json.dumps({
    'model': 'claude-3-5-sonnet-20240620',
    'inputSize': 200_000,
    'outputSize': 8192,
  }))) | {
    'endpoint': urllib.parse.urlparse(
      o.get('DEVAGENT_PROVIDER_ANTHROPIC_CHAT_ENDPOINT', 'https://api.anthropic.com/v1/messages')
    ),
    'token': get_chain(o,
      'DEVAGENT_PROVIDER_ANTHROPIC_CHAT_TOKEN',
      'DEVAGENT_PROVIDER_ANTHROPIC_TOKEN',
    ),
  }

def load_embed_config(**o: str) -> None:
  return None # Anthropic does not support Embeddings

def load_provider_config(**o: str) -> ProviderCfg:
  """Loads the Anthropic Provider Config"""
  return json.loads(o.get('DEVAGENT_PROVIDER_ANTHROPIC_CFG', json.dumps({
    'name': 'Anthropic',
  })))

@dataclass
class AnthropicProvider(ModelProvider):
  """Model Provider using the Anthropic Public API"""
  cfg: ProviderCfg

  ### Chat Interface ###

  def chat_req_headers(self) -> dict[str, str]:
    return { # API Defaults, but can be overridden
      'anthropic-version': '2023-06-01', # https://docs.anthropic.com/en/api/versioning
    } | self.cfg.get('httpHeaders', {}) | self.cfg['chat'].get('httpHeaders', {}) | {
      'x-api-key': self.cfg['chat']['token'],
    }
  def chat_req_body(self, *messages: Message, **kwargs) -> dict[str, Any]:
    if messages[0]['role'] == 'system':
      logger.debug('Anthropic Provider does not support system messages, will extract')
      system = messages[0]['content']
      messages = messages[1:]
    else:
      system = SYSTEM_PROMPT['content']

    ### Create the `Psuedo` System Prompt
    system_prompt: list[Message] = [
      { 'role': 'user', 'content': system },
    ]
    # NOTE: Anthropic requires user/assistant turn based messaging so only add the assistant response if the first message is from the user
    if messages[0]['role'] == 'user': system_prompt.append(
      { 'role': 'assistant', 'content': 'I understand, please continue.' }
    )
    return { # Model Defaults that can be overridden
      'max_tokens': self.cfg['chat']['outputSize'],
    } | self.cfg['chat'].get('props', {}) | { # API Ref: https://docs.anthropic.com/en/api/messages
      'model': self.cfg['chat']['model'],
      'system': SYSTEM_PROMPT['content'],
      'messages': [ *system_prompt, *messages ],
      'stream': False,
    }
  def chat_extract_content(self, body: bytes) -> list[bytes]:
    return json.loads(body)['content'][0]['text']
  def chat_extract_error(self, body: bytes) -> ProviderError:
    return { 'kind': 'error' } | json.loads(body)
