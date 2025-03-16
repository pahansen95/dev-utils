"""

Load Providers

"""

from .core import *

@cache
def _import_provider_loader(
  provider_name: Literal['OpenAI', 'AzureOAI', 'AzureAI', 'Anthropic'],
) -> p.ProviderLoader:
  if provider_name.lower() == 'OpenAI'.lower():
    from . import OpenAI as Provider
  elif provider_name.lower() == 'AzureOAI'.lower():
    from . import AzureOpenAI as Provider
  elif provider_name.lower() == 'AzureAI'.lower():
    raise NotImplementedError
  elif provider_name.lower() == 'Anthropic'.lower():
    raise NotImplementedError
  else: raise ValueError(provider_name)
  return Provider

def load_provider_from_env(
  provider_name: Literal['OpenAI', 'AzureOAI', 'AzureAI', 'Anthropic'],
  env: Mapping[str, str] = os.environ,
  partial_cfg: dict[str, str] = None,
) -> p.ModelProvider:
  provider_loader = _import_provider_loader(provider_name)
  return provider_loader.load_provider(env, env, partial_cfg)

__all__ = [
  'load_provider_from_env',
]