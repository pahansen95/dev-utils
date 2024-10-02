from __future__ import annotations
from typing import TypedDict, Any, Mapping
from collections.abc import Iterable
import time, subprocess, pathlib, json, re, logging

logger = logging.getLogger(__name__)

def get_chain(o: Mapping, *keys, default: Any = None,) -> Any | None:
  """Same as dict.get, but try multiple keys ordered as provided"""
  for k in keys:
    if (val := o.get(k)) is not None: return val
  return default

class Chat:
  """Semantic Type for a Chat Conversation"""

  class Message(TypedDict):
    publisher: str
    """Who authored the message"""
    content: str
    """The message content"""
    created: int
    """When the message was created as a Unix Nanosecond Timestamp"""

    @staticmethod
    def factory(
      publisher: str,
      content: str,
      created: int = None,
    ) -> Chat.Message:
      return {
        'publisher': publisher,
        'content': content,
        'created': created or time.time_ns(),
      }

  class Log(TypedDict):
    log: list[Chat.Message]

    @staticmethod
    def factory(
      msgs: Iterable[Chat.Message] = []
    ) -> Chat.Log:
      return { 'log': list(msgs) }

    @staticmethod
    def marshal(log: Chat.Log) -> bytes:
      """
      
      Marshal the Chat Log into a Human Readable Format:

      ````markdown
      <!-- LOG METADATA [JSON Obj] -->

      ---

      <!-- MSG METADATA [JSON Obj] -->

      [Message Content]

      ---
      ````

      The trailing `---` is stripped.
      
      """

      def _embed(obj: dict, kind: str) -> str: return f'<!-- {kind.upper()} METADATA {json.dumps(obj)} -->'

      return (
        '\n'.join([
          _embed({
            'size': len(log['log']),
          }, 'log'),
          *[
            '\n'.join([
              '\n---\n',
              _embed({k: v for k, v in msg.items() if k in {'publisher', 'created'}}, 'msg') + '\n',
              msg['content'].strip(),
            ]) for idx, msg in enumerate(log['log'])
          ],
        ]).strip() + '\n'
      ).encode()

    LOG_METADATA_RE = re.compile(r'<!-- LOG METADATA (.+) -->')
    MSG_METADATA_RE = re.compile(r'---\s+<!-- MSG METADATA (.+) -->')

    @staticmethod
    def unmarshal(data: bytes) -> 'Chat.Log':
      """
      Unmarshal the Chat Log from a Human Readable Format

      Reverses the marshal process, reconstructing the Chat.Log object
      from the given byte string.
      """
      text = data.decode().strip()

      # Extract metadata from the first line
      log_metadata_match = Chat.Log.LOG_METADATA_RE.match(text)
      if not log_metadata_match: raise ValueError('Missing Log Metadata')
      log_metadata = json.loads(log_metadata_match.group(1))
      log_size = log_metadata['size']
      text = text[log_metadata_match.end():]

      msg_metadata_matches = list(Chat.Log.MSG_METADATA_RE.finditer(text))
      if len(msg_metadata_matches) != log_size: raise ValueError(f'Mismatched Log Size: Got {len(msg_metadata_matches)}, Expected {log_size}')

      messages = []      
      for idx, match in enumerate(msg_metadata_matches):
        msg_metadata = json.loads(match.group(1))
        msg_start = match.end()
        if idx + 1 < len(msg_metadata_matches): msg_end = msg_metadata_matches[idx + 1].start()
        else: msg_end = len(text)
        msg_content = text[msg_start:msg_end].strip()
        messages.append({
          'publisher': msg_metadata['publisher'],
          'content': msg_content,
          'created': msg_metadata['created'],
        })

      return { 'log': messages }

class Git:
  """Semantic Type for basic Git Utility"""
  
  @staticmethod
  def sync(
    msg: str,
    pathspecs: list[str],
    worktree: pathlib.Path,
  ):
    subprocess.run(['git', 'add', *pathspecs], check=True, cwd=worktree.as_posix(), capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', msg, *pathspecs], check=True, cwd=worktree.as_posix(), capture_output=True, text=True)
