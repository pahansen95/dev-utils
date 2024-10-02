"""

Implements Various Model Providers for Azure's AI Service

NOTE:

- Azure AI provides multiple APIs based on the Model

"""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass, KW_ONLY, field
import urllib.parse, logging, json, functools, os, base64

### Pkg Imports
from ....Utils import get_chain
from .. import ModelProvider, ProviderCfg, ProviderError, ModelCfg
from ..chat import Message
from .OpenAI import OpenAIProvider
###

logger = logging.getLogger(__name__)

### Azure OpenAI

def load_oai_chat_config(**o: str) -> ModelCfg:
  try: chat_cfg = o['DEVAGENT_PROVIDER_AZUREOAI_CHAT_CFG'] 
  except KeyError:
    logger.critical('Azure OpenAI Requires an explicit Chat Model Config be declared')
    raise
  try: chat_endpoint = o['DEVAGENT_PROVIDER_AZUREOAI_CHAT_ENDPOINT'] 
  except KeyError:
    logger.critical('Azure OpenAI Requires an explicit Chat Model Endpoint be declared')
    raise

  return json.loads(chat_cfg) | {
    'endpoint': urllib.parse.urlparse(chat_endpoint),
    'token': get_chain(o,
      "DEVAGENT_PROVIDER_AZUREOAI_CHAT_TOKEN",
      "DEVAGENT_PROVIDER_AZUREOAI_TOKEN",
    ),
  }

def load_oai_embed_config(**o: str) -> ModelCfg:
  return json.loads(o.get('DEVAGENT_PROVIDER_AZUREOAI_EMBED_CFG', json.dumps({
    'model': 'text-embedding-3-large',
    'inputSize': 8192,
    'outputSize': 3072,
    'outputDType': 'float32',
  }))) | {
    'endpoint': urllib.parse.urlparse(
      o.get('DEVAGENT_PROVIDER_AZUREOAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
    ),
    'token': get_chain(o,
      "DEVAGENT_PROVIDER_AZUREOAI_EMBED_TOKEN",
      "DEVAGENT_PROVIDER_AZUREOAI_TOKEN",
    )
  }

def load_oai_provider_config(**o: str) -> ProviderCfg:
  return json.loads(o.get('DEVAGENT_PROVIDER_AZUREOAI_CFG', json.dumps({
    'name': 'AzureOpenAI',
  })))

@dataclass
class AzureOAIProvider(OpenAIProvider):
  """Model Provider using the AzureOAI API; Azure slightly modifies the OpenAI API"""
  cfg: ProviderCfg

  ### Embedding Interface ###

  def embed_req_headers(self) -> dict[str, str]:
    (oai_headers := super().embed_req_headers()).pop('Authorization')
    return oai_headers | {
      'api-key': self.cfg['embed']['token'],
    }
  def embed_req_body(self, *inputs, **kwargs) -> dict[str, Any]:
    (oai_body := super().embed_req_body(*inputs, **kwargs)).pop('model')
    return oai_body

  ### Chat Interface ###
  # NOTE: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-07-01-preview/inference.yaml#L367-L430
  # NOTE: https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-07-01-preview/inference.yaml#L2345-L2577
  
  def chat_req_body(self, *messages: Message, **kwargs) -> dict[str, Any]:
    (oai_body := super().chat_req_body(*messages, **kwargs)).pop('model')
    return oai_body
  def chat_req_headers(self) -> dict[str, str]:
    (oai_headers := super().chat_req_headers()).pop('Authorization')
    return oai_headers | {
      'api-key': self.cfg['chat']['token'],
    }

### Azure Serverless

## Phi

## Mistral