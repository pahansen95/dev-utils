from __future__ import annotations
from types import TracebackType
from typing import Any, Literal, TypedDict, Required, TypeVar, Generic, ContextManager, ParamSpec
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, fields, KW_ONLY
from functools import wraps

import logging, contextlib

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
  _: KW_ONLY
  res: R = field(init=False, default=None)

  def _handle_exc(self, exc_type: type[Exception], exc: Exception, tb: TracebackType) -> bool:
    if self.handle_exc is None: raise exc
    else: return self.handle_exc(exc_type, exc, tb)
  
  def _handle_result(self, res: R | type[Retry]) -> bool:
    if self.handle_result is None: return not ( res is Retry )
    else: return self.handle_result(res)

  def _handle(self,
    fn: Callable[P, R],
    *args, **kwargs
  ) -> bool:
    try: self.res = fn(*args, **kwargs)
    except Exception as e: return self._handle_exc(type(e), e, e.__traceback__)
    else: return self._handle_result(self.res)

  def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def _retry_fn(*args: P.args, **kwargs: P.kwargs) -> R:
      for idx in range(self.attempts):
        logger.debug(f'attempt {idx+1} of {self.attempts+1}')
        okay = self._handle(fn, *args, **kwargs)
        if okay: return self.res
      raise RuntimeError(f'Failed to complete after {self.attempts+1} total attempts')
    return _retry_fn

RETRY_T = type[Retry]