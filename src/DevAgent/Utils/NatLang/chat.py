"""

Implements an Language Model Chat Protocol

"""
from __future__ import annotations
from typing import TypedDict, Optional, Any, NamedTuple , Literal
from collections.abc import ByteString, Generator
from dataclasses import dataclass, KW_ONLY, field
from contextlib import contextmanager
import logging, struct, yaml, requests, re

### Pkg Imports
from . import _Interface, ModelProvider, DEFAULT_HEADERS, HTTPStatus
###

logger = logging.getLogger(__name__)

class Message(TypedDict):
  role: Literal['system', 'user', 'assistant']
  content: str

@dataclass
class ChatInterface(_Interface):
  provider: ModelProvider
  _: KW_ONLY
  conn: requests.Session | None = field(default=None)
  """Connection Pooling"""

  def chat(self, *msg: Message) -> str:
    """Prompt a Language Model returning the response"""
    with self._request(
      'POST', self.provider.cfg['chat']['endpoint'].geturl(),
      headers=DEFAULT_HEADERS | self.provider.chat_req_headers(),
      json=self.provider.chat_req_body(*msg, **self.provider.cfg['chat'].get('props', {})),
    ) as (status, resp):
      if status.major == 2: return self.provider.chat_extract_content(resp.content)
      elif status.major in {4, 5}:
        msg = f"Chat Failed: {status} {resp.reason}"
        if resp.content is not None: msg += f': {self.provider.chat_extract_error(resp.content)}'
        logger.error(msg)
        raise RuntimeError(msg)
      else: raise NotImplementedError(f"Unhandled HTTP Status: {status}")

SYSTEM_PROMPT_MSG: Message = {
  'role': 'system',
  'content': """
You are a critical thinker & engineer. You are: A) inquisitive, advance the user's thoughts; B) contemplative, consider how to respond; C) straightforward, use plain language & be succinct; D) intellectually honest, respond with precision & accuracy, avoid speculation and hearsay, vocalize knowledge gaps.

Before responding develop a strategy for your response, articulating: A) the context; B) user expectations; C) your relevant knowledge; D) key takeaways for the user; E) an outline prioritizing information by relevance.

It is imperative you start by including your articulation inside a `<meta hidden k=strat>` element; transparent communication builds user trust. Here is an example of how to format your response:

````markdown
<!-- User -->

How do I sort a list of numbers in Python?

<!-- Assistant -->

<meta hidden k=strat>
This is a basic programming task in Python; a simple solution is adequate; I know of the list.sort() method and the sorted() builtin. My response should convey the what, how & why for these options. I'll respond with: A) a salient answer, B) a descriptive example, C) supplementary information and, D) a leading question.
</meta>

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

CODE_BLOCK_RE = re.compile(r'```([^\n\s]*)\n([\s\S]*?)\n```', re.MULTILINE)
def extract_markdown_code_block(doc: str, kind: str) -> str:
  """Extract the contents of a Code Block from a Markdown Document."""
  match = CODE_BLOCK_RE.search(doc)
  if match is None: raise ValueError("No Code Block Found.")
  if match.group(1) != kind: raise ValueError(f"Code Block Kind Mismatch: {match.group(1)} != {kind}")
  return match.group(2)
