from __future__ import annotations
from types import TracebackType
from typing import Any, Literal, TypedDict, Required, TypeVar, Generic, ContextManager, ParamSpec
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, fields, KW_ONLY
from functools import wraps

import logging, time

from ..Protocols import intern as p

logger = logging.getLogger()

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
  opts: dict[str, Any]
  """Runtime Tuning Parameters"""

@dataclass
class Model(p.Model):
  name: str
  cfg: ModelCfg

@dataclass
class ModelProvider(p.ModelProvider):
  models: dict[str, Model]
  
@dataclass
class ProviderSession(p.ProviderSession):
  provider: ModelProvider

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