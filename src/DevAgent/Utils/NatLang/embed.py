"""

Implements an Embedding Protocol

"""
from __future__ import annotations
from typing import TypedDict, Optional, Any, NamedTuple
from collections.abc import ByteString, Generator
from dataclasses import dataclass, KW_ONLY, field
from contextlib import contextmanager
import logging, struct, yaml, requests

### Package Imports
from . import _Interface, ModelProvider, DEFAULT_HEADERS, HTTPStatus
###

logger = logging.getLogger(__name__)

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
class EmbeddingInterface(_Interface):
  provider: ModelProvider
  _: KW_ONLY
  conn: requests.Session | None = field(default=None)
  """Connection Pooling"""
  
  def embed(self, *item: str) -> Embedding:
    """Produce a batch of embeddings for each textual item"""
    logger.debug(f"Embedding w/ `{self.provider.embed_model_identifier}`")
    with self._request(
      'POST', self.provider.cfg['embed']['endpoint'].geturl(),
      headers=DEFAULT_HEADERS | self.provider.embed_req_headers(),
      json=self.provider.embed_req_body(*item, **self.provider.cfg['embed'].get('props', {})),
    ) as (status, resp):
      if status.major == 2: return Embedding.factory(
        *self.provider.embed_extract_content(resp.content),
        shape=(self.provider.cfg['embed']['outputSize'],),
        dtype=self.provider.cfg['embed']['outputDType'],
        model=self.provider.embed_model_identifier,
      )
      elif status.major in {4, 5}:
        msg = f"Embed Failed: {status} {resp.reason}"
        if resp.content is not None: msg += f': {self.provider.embed_extract_error(resp.content)}'
        logger.error(msg)
        raise RuntimeError(msg)
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
# INTERFACE = EmbeddingInterface.factory(BACKEND.factory)
