"""

Providers of Natural Lanugage Models

"""
from __future__ import annotations
from typing import Literal
import logging

logger = logging.getLogger(__name__)

from .. import ModelProvider

kind_t = Literal['chat', 'embed']

class UnknownProvider(NotImplementedError): ...
class ProviderLoadError(RuntimeError): ...

PROVIDER_REGISTRY: dict[str, ModelProvider] = {}
def add_provider_to_registry(provider: str, registry: dict[str, ModelProvider] = PROVIDER_REGISTRY, **o):
  assert provider not in registry

  if provider == 'OpenAI':
    from . import OpenAI
    load_provider_cfg = OpenAI.load_provider_config
    load_chat_cfg = OpenAI.load_chat_config
    load_embed_cfg = OpenAI.load_embed_config
    provider_factory = OpenAI.OpenAIProvider
    # chat_provider = OpenAI.OpenAIProvider(cfg=OpenAI.load_provider_config(**o) | { 'chat': OpenAI.load_chat_config(**o) })
  elif provider.startswith('Azure'):
    from . import Azure
    match provider.split('Azure', maxsplit=1)[-1]:
      case 'OAI':
        load_provider_cfg = Azure.load_oai_provider_config
        load_chat_cfg = Azure.load_oai_chat_config
        load_embed_cfg = Azure.load_oai_embed_config
        provider_factory = Azure.AzureOAIProvider
      case _: raise UnknownProvider(f'{provider} is not an available selection')
  elif provider == 'Anthropic':
    from . import Anthropic
    # provider = Anthropic.AnthropicProvider(cfg=Anthropic.load_config(**o))
    load_provider_cfg = Anthropic.load_provider_config
    load_chat_cfg = Anthropic.load_chat_config
    load_embed_cfg = Anthropic.load_embed_config
    provider_factory = Anthropic.AnthropicProvider
  else: raise UnknownProvider(f'{provider} is not an available selection')

  chat_cfg = None
  try: chat_cfg = load_chat_cfg(**o)
  except KeyError as e: logger.warning(f'Could not load Chat Config for Provider `{provider}`: {e}')
  embed_cfg = None
  try: embed_cfg = load_embed_cfg(**o)
  except KeyError as e: logger.warning(f'Could not load Embed Config for Provider `{provider}`: {e}')
  
  provider_cfg = load_provider_cfg(**o)
  if chat_cfg is not None: provider_cfg |= { 'chat': chat_cfg }
  if embed_cfg is not None: embed_cfg |= { 'embed': embed_cfg }

  registry[provider] = provider_factory(cfg=provider_cfg)

def get_provider_name_from_map(kind: kind_t, **o) -> str:
  assert kind in ('chat', 'embed')
  if (val := o.get(f'DEVAGENT_PROVIDER_{kind.upper()}', None)) is not None: return val
  if (val := o.get('DEVAGENT_PROVIDER', None)) is not None: return val
  logger.warning(f'Failed to load a {kind.capitalize()} Model Provider from: DEVAGENT_PROVIDER_{kind.upper()} or DEVAGENT_PROVIDER')
  raise KeyError(f'[ "DEVAGENT_PROVIDER_{kind.upper()}", "DEVAGENT_PROVIDER" ]')
  
def load_provider_by_name_from_map(provider_name: str, requires: set[kind_t], registry = PROVIDER_REGISTRY, **o: str) -> ModelProvider:
  """Load a provider from a Flat Mapping of Configurables into the provided registry."""

  if provider_name not in registry: add_provider_to_registry(provider_name, registry=registry, **o)

  if (missing := [
    r for r in requires
    if r not in registry[provider_name].cfg.keys()
  ]): raise ProviderLoadError( f'Provider Failed to load the following Configs: {missing}' )

  return registry[provider_name]
