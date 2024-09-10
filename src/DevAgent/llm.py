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
      system = msg[0]['content']
      msg = msg[1:]
    else:
      system = 'You are an Automated Python Peer Programmer.'
    if msg[0]['role'] != 'user':
      msg = [
        { 'role': 'user', 'content': "begin" }, # Anthropic requires the first Message be a user message
        *msg
      ]
    return { # API Ref: https://docs.anthropic.com/en/api/messages
      'model': MODEL_ID,
      **MODEL_CFG,
      'system': system,
      'messages': list(msg),
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
You are a helpful Peer Programming Assistant that the user is relying on while they work through their project.

Always first consult the provided context when
A) the user asks you a question,
B) the user requests you to implement a feature
or C) the user otherwise refers to the context or some generally related topic.

Do not assume to know everything, identify scenarios where there is a gap, such as when
A) you lack the knowledge to accurately respond to the user,
B) the user's intent is ambiguous
or C) you begin to hallucinate unverified information.
In such cases, first ask the user to help you; you can
A) request the user provide you with relevant information,
B) request the user to clarify their intent,
or C) ask the user relevant questions to help drive understanding.

When communicating with the user, make sure to be thoughtful in your response by
A) talking through complex concepts by first breaking it down into a series of simpler concepts,
B) approaching logical quandary through methodical analysis
or C) assuming the user is intelligient & will ask for further clarification if they don't understand something.

When interacting with the user, you should act like their professional peer by
A) speaking plainly & straightforward with transparent intent,
B) avoiding apologitic sentiments, such as "I aplologize" or "You are absolutely right",
and C) staying relevant to the topic at hand.
""".replace('\n\n', '\0').replace('\n', ' ').replace('\0', '\n').strip()
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