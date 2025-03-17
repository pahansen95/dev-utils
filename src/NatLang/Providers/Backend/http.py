"""

A HTTP based Backend Provider

"""

from .core import *
import requests

__all__ = [
  'RESP_T',
  'HTTPBackend',
  'ProviderHTTPBackendError',
]

class ProviderHTTPBackendError(ProviderBackendError): ...

@cache
def _url(base: str, *path: str) -> str: return '/'.join(s.strip('/') for s in (base, *path))

def default_headers(**kwargs) -> dict[str, str]: return {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  **kwargs
}

RESP_T = tuple[tuple[int, int], requests.Response]
@dataclass
class HTTPBackend:
  url: str
  _: KW_ONLY
  session: requests.Session = field(default_factory=requests.Session)
  auth_headers: Callable[[], dict[str, str]] = field(default_factory=lambda: {})
  headers: dict = field(default_factory=default_headers)

  def __enter__(self): return self
  def __exit__(self, *args): self.session.close()

  def _url(self, *path: str) -> str: return _url(self.url, *path)

  def _headers(self, **kwargs) -> dict[str, str]: return {
    **self.headers,
    **( self.auth_headers() ),
    **kwargs,
  }
  
  @contextmanager
  def request(self,
    path: str,
    body: dict,
    headers: dict[str, str] = None,
    params: dict[str, str | None] = None,
    raise_for_status = True,
  ) -> Generator[RESP_T, None, None]:
    """Generic request handler with retry logic"""
    _url = self._url(path)
    _headers = self._headers(**(headers or {}))
    with self.session.request(
      'POST', _url, json=body, headers=_headers, params=params,
    ) as resp:
      major, minor = tuple(divmod(resp.status_code, 100))
      if major in {4, 5}:
        try: err = json.dumps(resp.json(), indent=2)
        except:
          try: err = resp.text
          except: err = '[No Error Message Provided]'
        logger.debug(f'POST {_url}\n{err}')
        if raise_for_status: resp.raise_for_status()
      elif major == 2: logger.debug(f'POST {_url}\n{json.dumps(resp.json(), indent=2)}')
      else:
        logger.debug(f'POST {_url}\nStatus Code: {resp.status_code}')
        raise NotImplementedError
      yield (major, minor), resp

  @contextmanager
  def streaming_request(self,
    path: str,
    body: dict = None,
    headers: dict[str, str] = None,
    params: dict[str, str | None] = None,
    raise_for_status = True,
  ) -> Generator[Any, None, None]:
    raise NotImplementedError
  
  @Retry()
  def retry_request(self,
    path: str,
    body: dict,
    headers: dict[str, str] = None,
    params: dict[str, str | None] = None,
    raise_for_status = True,
  ) -> dict:
    """Method specifically for retryable requests"""
    with self.request(
      path, body, headers=headers, params=params, raise_for_status=raise_for_status
    ) as (
      (major, _),
      resp,
    ):
      if major in { 5 }: return Retry
      elif major in { 4 }: raise ProviderHTTPBackendError( obj=( { 'kind': 'error' } | resp.json() ) )
      assert major in { 2 }
      return { 'kind': 'response' } | resp.json()
