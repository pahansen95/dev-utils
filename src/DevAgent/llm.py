from __future__ import annotations
import http.client, json, logging, os, re, time, threading, weakref, ssl
from urllib.parse import urlparse
from typing import TypedDict, Literal, Any
from collections.abc import Callable, Generator
from contextlib import contextmanager

from ._utils.http import ConnectionProxy

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json; charset=utf-8',
  'Connection': 'keep-alive',
  'Keep-Alive': 'timeout=900', # Keep alive for at least 15m
}
PROVIDER = os.environ.get('DEVAGENT_CHAT_PROVIDER', 'openai')
if PROVIDER == 'azure-openai':
  url = urlparse(os.environ['AZURE_OPENAI_ENDPOINT'])
  assert url.path.endswith('/chat/completions')
  MODEL_ID = url.path.split('/')[-3] # The third from the end is the Azure model deployment
  MODEL_CFG = { k: v for k, v in os.environ.get('AZURE_OPENAI_MODEL_CFG', { 'max_tokens': int(16 * 1024) }).items() if k in { 'max_tokens', } }
  assert url.scheme == 'https'
  def _connection_factory() -> http.client.HTTPConnection: return http.client.HTTPSConnection(url.hostname, port=url.port or 443)
  def _load_headers() -> dict[str, str]: return DEFAULT_HEADERS | {
    'api-key': os.environ['AZURE_OPENAI_API_KEY'],
  }
  def _load_body(*msg: Message) -> dict[str, Any]: return { # For Spec see https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-07-01-preview/inference.yaml
    **MODEL_CFG,
    'messages': list(msg),
    'stream': False,
  }
  def extract_content(body: bytes) -> str: return json.loads(body)['choices'][0]['message']['content']
elif PROVIDER == 'openai':
  MODEL_ID = os.environ.get('OPENAI_MODEL_ID', 'gpt-4o-mini-2024-07-18') # 'gpt-4o-2024-08-06'
  MODEL_CFG = { k: v for k, v in os.environ.get('OPENAI_MODEL_CFG', { 'max_tokens': int(16 * 1024) }).items() if k in { 'max_tokens', } }
  url = urlparse(os.environ['OPENAI_ENDPOINT'])
  assert url.scheme == 'https'
  def _connection_factory() -> http.client.HTTPConnection: return http.client.HTTPSConnection(url.hostname, port=url.port or 443)
  def _load_headers() -> dict[str, str]: return DEFAULT_HEADERS | {
    'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
  }
  def _load_body(*msg: Message) -> dict[str, Any]: return {
    'model': MODEL_ID,
    **MODEL_CFG,
    'messages': list(msg),
    'stream': False,
  }
  def extract_content(body: bytes) -> str: return json.loads(body)['choices'][0]['message']['content']
elif PROVIDER == 'anthropic':
  url = urlparse(os.environ['ANTHROPIC_ENDPOINT'])
  assert url.scheme == 'https'
  MODEL_ID = os.environ.get('ANTHROPIC_MODEL_ID', 'claude-3-5-sonnet-20240620') # https://docs.anthropic.com/en/api/versioning
  MODEL_CFG = { k: v for k, v in os.environ.get('ANTHROPIC_MODEL_CFG', { 'max_tokens': int(8 * 1024) }).items() if k in { 'max_tokens', } }
  def _connection_factory() -> http.client.HTTPConnection: return http.client.HTTPSConnection(url.hostname, port=url.port or 443)
  def _load_headers() -> dict[str, str]: return DEFAULT_HEADERS | {
    'x-api-key': os.environ["ANTHROPIC_API_KEY"],
    'anthropic-version': '2023-06-01', # https://docs.anthropic.com/en/api/versioning
  }
  def _load_body(*msg: Message) -> dict[str, Any]:
    if msg[0]['role'] == 'system':
      logger.debug('Anthropic Provider does not support system messages, will extract')
      system = msg[0]['content']
      msg = msg[1:]
    else:
      system = 'You are a helpful assistant, follow my instructions.'

    ### Create the `Psuedo` System Prompt
    role_msgs: list[Message] = [
      { 'role': 'user', 'content': system },
    ]
    # NOTE: Anthropic requires user/assistant turn based messaging so only add the assistant response if the first message is from the user
    if msg[0]['role'] == 'user': role_msgs.append(
      { 'role': 'assistant', 'content': 'I will assume the role & characteristics you have given me, starting now.' }
    )
    return { # API Ref: https://docs.anthropic.com/en/api/messages
      'model': MODEL_ID,
      **MODEL_CFG,
      'system': 'Assume the role & characteristics the user provides you.',
      'messages': [ *role_msgs, *msg ],
      'stream': False,
    }
  def extract_content(body: bytes) -> str: return json.loads(body)['content'][0]['text']
elif PROVIDER == 'ollama':
  url = urlparse(os.environ['OLLAMA_ENDPOINT'])
  assert url.scheme in {'http', 'https'}
  MODEL_ID = os.environ.get('OLLAMA_MODEL_ID', 'phi3.5:3.8b-mini-instruct-q8_0')
  MODEL_CFG = { } # No Model Configuration for OLLAMA
  def _connection_factory() -> http.client.HTTPConnection:
    if url.scheme == 'http': return http.client.HTTPConnection(url.hostname, port=url.port or 80)
    elif url.scheme == 'https': return http.client.HTTPSConnection(url.hostname, port=url.port or 443)
    else: raise RuntimeError("Unsupported Scheme for OLLAMA_ENDPOINT.")
  def _load_headers() -> dict[str, str]: return DEFAULT_HEADERS | {
    # ...
  }
  def _load_body(*msg: Message) -> dict[str, Any]: return { # See https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    'model': MODEL_ID,
    **MODEL_CFG,
    'messages': list(msg),
    'stream': False,
  }

  def extract_content(body: bytes) -> str: return json.loads(body)['message']['content']
else: raise RuntimeError(f'Uknown Provider: {PROVIDER}')

# The Module Connection; Will clean up the Connection on Module GC
conn = ConnectionProxy(_connection_factory)

@contextmanager
def chat_completion(*msg: Message) -> Generator[http.client.HTTPResponse, None, None]:
  logger.debug(f"Using {PROVIDER} Model: {MODEL_ID} @ {url.geturl()} w/ Config {MODEL_CFG}")
  with conn.lock:
    conn.request('POST', url.geturl(), headers=_load_headers(), body=json.dumps(_load_body(*msg)))
    try: yield conn.getresponse()
    finally: pass # Don't close the Response, that will also close the Connection

class Message(TypedDict):
  """A message to be sent to the LLM."""
  role: Literal['system', 'user', 'assistant']
  content: str
SYSTEM_PROMPT_MSG: Message = {
  'role': 'system',
  'content': """
You are a critical thinker & engineer. You are: A) inquisitive, augmenting the user's Chain of Though; B) contemplative, considering how you respond ; C) straightforward, using plain language & being succinct; D) intellectually honest, responding with precise & accurate information, avoiding speculation and hearsay and vocalizing your knowledge gaps.

Before responding, think out loud, strategizing your response: A) contemplate the context; B) consider user expectations; C) identify your knowledge; D) articulate key takeaways; E) outline your response prioritizing information by relevance. Separate your thoughts from your response, here is an example:

````markdown
<!-- User -->

How do I sort a list of numbers in Python?

<!-- Assistant -->

```xml
<meta kind=thought>
This is a basic programming task in Python; a simple solution is adequate; I know of the list.sort() method and the sorted() builtin. My response should convey the what, how & why for these options. I'll respond with: A) a salient answer, B) a descriptive example, C) supplementary information and, D) a leading question.
</meta>
```

You can use `list.sort()` or the `sorted()` builtin.

```python
unsorted = [3, 1, 2]
# sorted() returns a sorted copy of the list
assert sorted(unsorted) is not unsorted
# list.sort() sorts the list mutating it inplace
print(unsorted.sort)
```

`list.sort()` is faster for large lists. `sorted()` ensures immutability. Do you need more details on sorting options or performance?
````

Generally follow these guidelines when conversing:

- Always consult the context & prioritize it's information.
- Assume the user is intelligient & will ask you to clarify if necessary.
- Approach logical quandary through methodical analysis.
- Avoid hallucinating knowledge but identify when you do & inform the user.
- Speak in active voice
- Exclude statements that are apologetic or gratify the user; you are their peer.
- Stay relevant to the topics at hand.
""".strip()
}

def chat(*msg: Message) -> str:
  """Prompt an LLM instance with a given prompt returning the response."""
  while True:
    with chat_completion(*msg) as resp:
      resp_data = resp.read()
      if resp.status in { 200, 201 }: break
      elif resp.status in { 500, 502 }:
        logger.warning(f"LLM Server Errror; will retry...\n{resp_data}")
        time.sleep(1)
        continue
      else:
        logger.error(f"Unhandled LLM Error...\n{resp.status}: {resp_data}")
        raise RuntimeError(f"LLM Error: {resp_data}")
  content = extract_content(resp_data)
  assert isinstance(content, str)
  logger.debug(f"LLM Response...\n{content}")
  return content

CODE_BLOCK_RE = re.compile(r'```([^\n\s]*)\n([\s\S]*?)\n```', re.MULTILINE)
def extract_markdown_code_block(doc: str, kind: str) -> str:
  """Extract the contents of a Code Block from a Markdown Document."""
  match = CODE_BLOCK_RE.search(doc)
  if match is None: raise ValueError("No Code Block Found.")
  if match.group(1) != kind: raise ValueError(f"Code Block Kind Mismatch: {match.group(1)} != {kind}")
  return match.group(2)