"""

Implements Backends for Text Embedding

TODO: Refactor the Client & Backend Implementations

"""
from __future__ import annotations
from dataclasses import dataclass, KW_ONLY, field
from typing import TypedDict, Optional, Any, NamedTuple
from collections.abc import Iterable, Callable, ByteString
import http.client, urllib.parse, os, json, logging, ssl, threading, weakref, base64, yaml, struct, math

from .._utils.http import ConnectionProxy, HTTPStatus, HTTPConnection, HTTPResponse

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json; charset=utf-8',
  'Connection': 'keep-alive',
  'Keep-Alive': 'timeout=900', # Keep alive for at least 15m
}
@dataclass
class Backend:
  cfg: Backend.Cfg

  class Cfg(TypedDict):
    url: urllib.parse.ParseResult
    model: str
    modelCfg: dict[str, Any]
    httpHeaders: dict[str, str]
  
  @classmethod
  def factory(cls) -> Backend:
    """Load the Embeddings Interface"""
    raise NotImplementedError
  
  @property
  def fully_qualified_name(self) -> str:
    """Returns the Fully Qualified Name of the Model"""
    return f'{type(self).__name__.strip('_')}:{self.cfg['model']}'

  def embed_req_body(self, *args, **kwargs) -> dict[str, Any]:
    """Construct a Request Body"""
    raise NotImplementedError
  
  def embed_extract_content(self, body: bytes) -> list[bytes]:
    """Extract the Content from the Response"""
    raise NotImplementedError

### OpenAI Embeddings

@dataclass
class _OpenAI(Backend):
  cfg: _OpenAI.Cfg

  @classmethod
  def factory(cls) -> _OpenAI:
    """Load the Embeddings Interface for Open AI"""
    output: Backend.Cfg = {
      'url': urllib.parse.urlparse(os.environ.get('OPENAI_EMBED_ENDPOINT', 'https://api.openai.com/v1/embeddings')),
      'model': os.environ.get('OPENAI_EMBED_MODEL', 'text-embedding-3-large'),
      'modelCfg': {
        'output_dims': 3072, # See https://platform.openai.com/docs/models/embeddings
        'output_dtype': 'float32',
        'input_size': 8192, # No Authorative Reference; only normative comments
      } | json.loads(os.environ.get('OPENAI_EMBED_MODEL_CFG', r'{}')),
      'httpHeaders': {
        'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
      }
    }
    return cls(cfg=output)

  def embed_req_body(self, *inputs: str) -> dict[str, Any]:
    return {
      'input': list(inputs),
      'model': self.cfg['model'],
      'dimensions': self.cfg['modelCfg']['output_dims'],
      'encoding_format': 'base64',
    }

  def embed_extract_content(self, body: bytes) -> list[bytes]:
    """Returns the List of Embeddings as Bytes"""
    content = json.loads(body)
    # logger.debug(f"Extracted Content: {content}")
    return list(map(base64.b64decode, (d['embedding'] for d in content["data"])))

@dataclass
class _AzureOpenAI(_OpenAI):
  cfg: _AzureOpenAI.Cfg

  @classmethod
  def factory(cls) -> _AzureOpenAI:
    """Load the Embeddings Interface for Open AI"""
    endpoint = urllib.parse.urlparse(os.environ['AZURE_OPENAI_EMBED_ENDPOINT'])
    output: Backend.Cfg = {
      'url': endpoint,
      'model': endpoint.path.split('/deployments/')[1].split('/')[0],
      'modelCfg': {
        'output_dims': 3072, # See https://platform.openai.com/docs/models/embeddings
        'output_dtype': 'float32',
        'input_size': 8192, # No Authorative Reference; only normative comments
      } | json.loads(os.environ.get('AZURE_OPENAI_EMBED_MODEL_CFG', r'{}')),
      'httpHeaders': {
        'api-key': os.environ['AZURE_OPENAI_API_KEY'],
      }
    }
    return cls(cfg=output)

PROVIDER = os.environ.get('DEVAGENT_EMBED_PROVIDER', 'openai')
if PROVIDER == 'openai': BACKEND: type[Backend] = _OpenAI
elif PROVIDER == 'azure-openai': BACKEND = _AzureOpenAI
else: logger.warning(f"Unknown Embedding Provider: {PROVIDER}")

### Embedding Interface

class Embedding(TypedDict):
  """A Single or Batch of Text Embeddings"""
  buffer: bytes
  """The Raw Data of the Embeddings"""
  shape: tuple[int, ...]
  """The Embedding's Shape"""
  dtype: str
  """The Data's Type"""
  model: str
  """The Unique Identifier that generated this Embedding"""
  metadata: Optional[dict[str, Any]]
  """Optional Metadata associated w/ this Embedding"""

  @staticmethod
  def factory(*embeds: ByteString, **kwargs) -> Embedding:
    """Creates an Embedding Batch from those provided. Assumes each embedding in `embeds` is of the same shape & dtype"""
    # TODO: Make this smarter; maybe just use Numpy
    assert len(kwargs['shape']) == 1 # Only supports Vectors currently
    dims = kwargs['shape'][0]
    assert kwargs['dtype'] in _VEC_DTYPE
    bytesize, _ = _VEC_DTYPE[kwargs['dtype']]
    assert all(len(e) == len(embeds[0]) for e in embeds)
    _dims, err = divmod(len(embeds[0]), bytesize)
    assert not err and _dims == dims, f'{dims=}, {_dims=}, {err}'
    return {
      'buffer': b''.join(embeds),
      'shape': (len(embeds), *kwargs['shape']),
      'dtype': kwargs['dtype'], # Required
      'model': kwargs['model'], # Required
    } | (
      { 'metadata': kwargs['metadata'] } if 'metadata' in kwargs else {}
    )

  @staticmethod
  def unpack(embedding: Embedding) -> list[_vector_t]:
    """Reference Implementation to unpack a Batched Embedding into a list of float vectors"""
    dtype = _VEC_DTYPE[embedding['dtype']]
    assert len(embedding['shape']) == 2 # Only support Batched Vectors
    batch_size, vec_size = embedding['shape']
    vec_bytesize = vec_size * dtype.bytesize
    _batch_size, err = divmod(len(embedding['buffer']), vec_bytesize)
    assert not err, 'Malformed Embedding'
    assert batch_size == _batch_size, f'Expected {batch_size} but got {_batch_size}'
    format = f"{vec_size}{dtype.format}"
    buffer = memoryview(embedding['buffer'])
    return list(
      struct.unpack(format, buffer[i:i+vec_bytesize])
      for i in range(0, len(buffer), vec_bytesize)
    )
  
  @staticmethod
  def marshal(embedding: Embedding) -> bytes:
    """Marshal the Embedding ommitting the actual embedding buffer; you can just grab that directly"""
    return yaml.safe_dump({
      'kind': Embedding.__name__,
      'spec': { k: v for k, v in embedding.items() if k in { 'dtype', 'model', 'metadata' } } | {
        'shape': list(embedding['shape']),
      }
    }, sort_keys=False).encode()
  
  @staticmethod
  def unmarshal(spec: bytes, buffer: bytes) -> Embedding:
    """Unmarshal the Embedding; injecting the raw buffer"""
    o = yaml.safe_load(spec)
    if o['kind'] != Embedding.__name__: raise ValueError(f'Uknown Kind: {o["kind"]}')
    return {
      'buffer': buffer,
      **(o['spec'] | {
        'shape': tuple(o['spec']['shape'])
      })
    }

  # @staticmethod
  # def slice(embedding: Embedding, start: int, stop: int) -> Embedding:
  #   """Slice an Embedding"""
  #   start, stop = max(0, start), min(embedding['shape'][0], stop)
  #   logger.debug(f'Slicing Embedding from [{start}, {stop}]')
  #   metadata = {}
  #   if ''
  #   return Embedding.factory(
  #     iter(...),
  #     shape=tuple(embedding['shape'][1:]),
  #     dtype=embedding['dtype'],
  #     model=embedding['model'],
  #     metadata={
  #       'DocChunks': embedding['metadata']['DocChunks'][start:stop],
  #       'Similarity': embedding['metadata']['Similarity'][start:stop],
  #     }
  #   )

@dataclass
class EmbeddingInterface:
  conn: ConnectionProxy
  _: KW_ONLY
  backend: Backend

  @classmethod
  def factory(cls, backend_factory: Callable[..., Backend]) -> EmbeddingInterface:
    backend = backend_factory()
    return cls(
      conn=ConnectionProxy(lambda: http.client.HTTPSConnection(
        backend.cfg['url'].hostname,
        port=backend.cfg['url'].port or (443 if backend.cfg['url'].scheme == 'https' else 80)
      )),
      backend=backend,
    )

  def embed(self, *item: str) -> Embedding:
    """Produce a batch of embeddings for each textual item"""
    logger.debug(f"Embedding w/ `{self.backend.fully_qualified_name}`")
    with self.conn.lock:
      self.conn.safe_request(
        'POST',
        self.backend.cfg['url'].geturl(),
        headers=DEFAULT_HEADERS | self.backend.cfg['httpHeaders'],
        body=json.dumps(self.backend.embed_req_body(*item)),
      )
      resp: HTTPResponse = self.conn.getresponse()
      # logger.debug(f'Embedding HTTP Headers...\n{json.dumps(dict(resp.getheaders()), indent=2)}')
      status = HTTPStatus.factory(resp.status)
      embed_vecs = self.backend.embed_extract_content(resp.read())
      if status.major == 2: return Embedding.factory(
        *embed_vecs,
        shape=(self.backend.cfg['modelCfg']['output_dims'],),
        dtype=self.backend.cfg['modelCfg']['output_dtype'],
        model=self.backend.fully_qualified_name,
      )
      elif status.major in {4, 5}:
        logger.error(f"Failed to Embed Text: {resp.status} {resp.reason}")
      else: raise NotImplementedError(f"Unhandled HTTP Status: {status}")

### NOTE: Temporary Embedding Vector Helpers
_vector_t = tuple[float, ...]
class _DTYPE(NamedTuple):
  bytesize: int
  format: str
_VEC_DTYPE: dict[str, _DTYPE] = {
  'float64': _DTYPE(8, 'd'),
  'float32': _DTYPE(4, 'f'),
  'float16': _DTYPE(2, 'e')
}
###
INTERFACE = EmbeddingInterface.factory(BACKEND.factory)
