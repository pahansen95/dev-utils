"""

HTTP Utilities for the Library

"""
from __future__ import annotations
import http.client, json, logging, os, re, time, threading, weakref, ssl
from urllib.parse import urlparse
from typing import TypedDict, Literal, Any, NamedTuple
from collections.abc import Callable, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

HTTPConnection = http.client.HTTPConnection | http.client.HTTPSConnection
HTTPResponse = http.client.HTTPResponse

class HTTPStatus(NamedTuple):
  major: int
  minor: int

  @classmethod
  def factory(cls, status: int) -> HTTPStatus:
    assert status >= 100 and status < 600
    return cls(*divmod(status, 100))

class ConnectionProxy:
  """A Proxy Object that wraps an HTTP Client Connection Object"""
  def __init__(self, connection_factory: Callable[[], http.client.HTTPConnection]):
    self._conn_factory = connection_factory
    self._conn: http.client.HTTPConnection | None = None
    self.lock = threading.Lock()
    self._finalize: Callable | None = None
    
  def _open_connection(self):
    if self._conn is not None:
      assert self._finalize is not None
      self._finalize()
    conn = self._conn_fact()
    def _close():
      try: conn.close()
      except: pass
    self._conn = conn
    self._finalize = weakref.finalize(self, _close)
  
  def _reset_connection(self):
    try: self._conn.close()
    except: pass
    finally: self._conn = self._conn_factory()

  def safe_request(self, *args, **kwargs):
    """Attempt to make a request, if it fails, retry"""
    ### TODO
    # if self._conn is None: self._conn = self._conn_factory()
    """NOTE
    I need to figure out how to better use Python's HTTP Library
    I'm leaking Disconnects when retrieving the response.
    Workaround for now is just to reset the connection on each request.
    """
    self._reset_connection()
    ###
    for _ in range(2):
      try:
        self._conn.request(*args, **kwargs)
        return
      except (ssl.SSLError, ConnectionError, OSError) as e:
        logger.debug(f"Attempting to reset connection: {type(e).__module__}.{type(e).__name__}({e})")
        self._reset_connection()
        continue
      except:
        logger.exception(f'Unhandled Exception in HTTP Request; Fatal')
        raise
    raise RuntimeError('Failed to make HTTP Request')

  def __getattr__(self, name: str) -> http.client.Any:
    """The Proxy Object forwards any unknown attribute to the underlying Connection Object"""
    if name == 'request': return getattr(self, 'safe_request')
    else: return getattr(self._conn, name)

  # def __getattribute__(self, name: str) -> http.client.Any:
  #   if (attr := getattr(self, name, None)) is not None: return attr
  #   elif name == 'request': return getattr(self, 'safe_request')
  #   else: return getattr(self._conn, name)
