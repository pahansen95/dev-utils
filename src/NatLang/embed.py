"""

Text Embeddings

"""

from .core import *

class DType(enum.StrEnum):
  u8 = 'u8'
  u16 = 'u16'
  u32 = 'u32'
  u64 = 'u64'
  f16 = 'f16'
  f32 = 'f32'
  f64 = 'f64'

class Embedding(TypedDict):
  buffer: ByteString
  """The batch of Latents; usually a continuous buffer of memory"""
  shape: tuple[int, ...]
  """The shape of the batch of latents; expected to be (batch_size, *latent_dimensions)"""
  dtype: DType
  """The Data Type of the buffer; ex. f32 or u8"""
  props: NotRequired[dict[str, Any]]
