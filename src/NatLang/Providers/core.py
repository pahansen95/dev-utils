from __future__ import annotations
from ..core import *
from dataclasses import dataclass, field, fields, KW_ONLY
from functools import wraps, cache
from contextlib import contextmanager

import logging, time, os

from abc import ABC

from ..Protocols import intern as p
from ..core import ModelCapabilities, model_capabilities
from .. import chat as c #, embed as e

logger = logging.getLogger()

@dataclass
class ProviderError(RuntimeError):
  obj: dict
  """The Error Object; JSON Encodable"""
  def __str__(self) -> str: return json.dumps(self.obj, separators=(':',','))

### Helpers

P = ParamSpec('P')
R = TypeVar('R')
@dataclass
class Retry:
  handle_exc: Callable[[type[Exception], Exception, TracebackType], bool] = field(default=None)
  handle_result: Callable[[R], bool] = field(default=None)
  attempts: int = 3
  backoff: Callable[[int], float] = field(default=None)

  def _handle_exc(self, exc_type: type[Exception], exc: Exception, tb: TracebackType) -> bool:
    if self.handle_exc is None: raise exc
    else: return self.handle_exc(exc_type, exc, tb)
  
  def _handle_result(self, res: R | type[Retry]) -> bool:
    if self.handle_result is None: return not ( res is Retry )
    else: return self.handle_result(res)
  
  def _backoff(self, idx: int) -> float:
    try:
      if self.backoff is None: return ( 0, 0.05, 0.1, 0.5, 1 )[idx]
      else: return self.backoff(idx)
    except IndexError:
      logger.warning(f'couldnt get a backoff value in seconds for retry {idx+1}; returning default of 1s')
      return 1

  def _eval(self,
    fn: Callable[P, R],
    *args, **kwargs
  ) -> tuple[bool, R | None]:
    try: res = fn(*args, **kwargs)
    except Exception as e: return self._handle_exc(type(e), e, e.__traceback__), None
    else: return self._handle_result(res), res

  def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def _retry_fn(*args: P.args, **kwargs: P.kwargs) -> R:
      for idx in range(self.attempts):
        logger.debug(f'fn[`{fn.__name__}`] evaluation attempt {idx+1} of {self.attempts+1}')
        ok, res = self._eval(fn, *args, **kwargs)
        if ok: return res
        else: time.sleep(self._backoff(idx)) # TODO: There's probably better things to do than block the thread
      raise RuntimeError(f'fn[`{fn.__name__}`] failed evaluation after {self.attempts+1} total attempts')
    return _retry_fn

RETRY_T = type[Retry]

NO_DEFAULT = type('NO_DEFAULT')
def load_env(*keys: str, env: Mapping[str, str] = os.environ, default: str | None = NO_DEFAULT,) -> str:
  v = NO_DEFAULT
  for k in keys:
    if k in env:
      v = env[k]
      break
  else:
    if default is NO_DEFAULT: raise KeyError(*keys)
    v = default
  return v
    

@dataclass
class BaseModelProvider(p.ModelProvider):

  @contextmanager
  def tune(self, model: str, **opts) -> Generator[None, None, None]:
    """Temporarily Applies tuning options to the model"""
    old_opts = self.models[model].opts
    try:
      self.models[model].opts |= opts
      yield
    finally:
      self.models[model].opts = old_opts
    
  def supports(self, model: str, capability: str) -> bool: return self.models[model].caps[capability]

# p.ModelCfg
class ModelCfg(TypedDict, total=False):
  version: Required[str]
  """The fully qualified versioned name identifying this model in the Provider API"""
  inputSize: Required[int]
  """The total allowed token input"""
  inputDType: str
  """Expected datatype of the input"""
  outputSize: Required[int]
  """The maximum allowed token output"""
  outputDType: str
  """Expected datatype of the output"""

@dataclass
class ChatModel(p.Model):
  name: str
  cfg: ModelCfg
  _: KW_ONLY
  opts: dict[str, Any] = field(default_factory=dict)
  caps: ModelCapabilities = field(init=False, default_factory=lambda: model_capabilities(chat=True))

@dataclass
class EmbedModel(p.Model):
  name: str
  cfg: ModelCfg
  _: KW_ONLY
  opts: dict[str, Any] = field(default_factory=dict)
  caps: ModelCapabilities = field(init=False, default_factory=lambda: model_capabilities(embed=True))
