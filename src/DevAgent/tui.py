"""

A TUI (Text User Interface) for a ChatLog

For now, this module uses the `micro` Text Editor to provide a Rich Text Editor.

"""
from __future__ import annotations
from typing import Literal
from collections.abc import Generator, Callable
import os, sys, tempfile, subprocess, shlex, re, logging

from .Utils import Chat

logger = logging.getLogger(__name__)

MICRO_BINDINGS = {
  "Alt-/": "lua:comment.comment",
  "CtrlUnderscore": "lua:comment.comment",
  "Ctrl-e": "EndOfLine",
  "Ctrl-a": "StartOfLine",
  "Ctrl-p": "CommandMode",
  "Ctrl-q": "QuitAll",
  "Ctrl-w": "Quit",
  "Ctrl-k": "SelectToEndOfLine",
  "Ctrl-u": "SelectToStartOfLine"
}
MICRO_SETTINGS = {
  "ft:markdown": { "softwrap": True },
  "tabsize": 2,
  "tabstospaces": True,
  "wordwrap": True
}

### Check if micro is installed
editor = os.environ.get('DEVAGENT_MICRO_BIN', 'micro')
try: subprocess.run(shlex.split(f'{editor} --version'), capture_output=False, check=True)
except FileNotFoundError: raise RuntimeError(f'`micro` is not installed or not in the PATH. Set the `DEVAGENT_MICRO_BIN` Environment Variable to the path of the `micro` binary.')

_editor_sentinel = '''
<!-- please enter your prompt below this line -->
'''.strip()
_editor_sentinel_re = re.compile(r'<!-- please enter your prompt below this line -->')
def read_markdown_from_editor(header: str = '') -> str:
  """Read Markdown Input from the User using the user's preferred Editor"""

  ### Get the User's Preferred Editor
  with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as tf:
    tf.write(''.join([
      '<!-- Scroll to the bottom to add a new message -->\n\n' if header else '',
      header,
      '\n\n',
      _editor_sentinel,
      '\n\n',
    ]).strip())
    tf.flush()
    subprocess.run(shlex.split(f'{editor} {tf.name}'), stdin=sys.stdin, stdout=sys.stdout, check=True)
    tf.seek(0)
    content = tf.read()
    start_idx = list(_editor_sentinel_re.finditer(content))[-1].end()
    if start_idx > len(content) or start_idx < 0: return ''
    else: return content[start_idx:].strip()

def preview_markdown_in_editor(content: str) -> None:
  """Preview a Markdown Message in the User's Preferred Editor"""
  with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as tf:
    tf.write( '<!-- RESPONSE PREVIEW - EDITS TO THIS FILE ARE NOT SAVED -->\n\n' + content )
    tf.flush()
    subprocess.run(shlex.split(f'{editor} {tf.name}'), stdin=sys.stdin, stdout=sys.stdout, check=True)

USER_MSG_SENTINEL = '<!-- Edit User Message Below -->'
user_msg_re = re.compile(r'<!-- Edit User Message Below -->\s*(.*)', re.DOTALL)
def run_editor(
  chat_log: str = '',
  chat_log_pos: tuple[int, int] = (1, 1),
  user_msg: str = '',
  split: Literal['horizontal', 'vertical'] = 'horizontal',
) -> str:
  """Run the Editor, splitting the Chat Log & User Message into seperate panes"""
  assert split in ('horizontal', 'vertical')

  with (
    tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as chat_log_tf,
    tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as user_msg_tf,
  ):
    if not chat_log: chat_log = '<!-- No Chat Log -->\n'
    chat_log_tf.write(chat_log)
    user_msg = USER_MSG_SENTINEL + '\n\n' + user_msg
    user_msg_tf.write(user_msg)

    # Get the Last Line & Column of the Chat Log & User Message
    # chat_log_row = chat_log.count('\n') + 1
    # chat_log_col = len(chat_log.rsplit('\n', maxsplit=1)[-1])
    # user_msg_row = user_msg.count('\n') + 1
    # user_msg_col = len(user_msg.rsplit('\n', maxsplit=1)[-1])
    user_msg_pos = (user_msg.count('\n') + 1, len(user_msg.rsplit('\n', maxsplit=1)[-1]))

    chat_log_cursor = ":".join([chat_log_tf.name, *map(str,chat_log_pos)])
    user_msg_cursor = ":".join([user_msg_tf.name, *map(str,user_msg_pos)])

    chat_log_tf.flush()
    user_msg_tf.flush()
    cmd = f'{editor} -softwrap true -wordwrap true -multiopen {split[0]}split -parsecursor true {chat_log_cursor} {user_msg_cursor}'
    # cmd = f'{editor} -softwrap true -wordwrap true -multiopen {split[0]}split -parsecursor true {chat_log_tf.name}:{chat_log_row}:{chat_log_col} {user_msg_tf.name}:{user_msg_row}:{user_msg_col}'
    logger.debug(f'Running Editor Command: {cmd}')
    subprocess.run(
      shlex.split(cmd),
      stdin=sys.stdin, stdout=sys.stdout, check=True,
    )

    user_msg_tf.seek(0)
    if (txt := user_msg_re.search(user_msg_tf.read())) is None: return ''
    else: return txt.group(1).strip()

def editor_loop(
  chat_log_factory: Callable[[], Chat.Log | None],
  user_msg_factory: Callable[[], Chat.Message | None],
  split: Literal['horizontal', 'vertical'] = 'horizontal',
) -> Generator[str, None, None]:
  """Run the Editor Loop"""
  while True:
    chat_log = chat_log_factory()
    user_msg = user_msg_factory()

    if chat_log is not None and chat_log['log']:
      chat_log_txt = '\n\n'.join(
        f"<!-- {msg['publisher']} -->\n\n{msg['content'].strip()}"
        for msg in chat_log['log']
      )
      ### Get the index of the last message in the chat log
      chat_log_idx = chat_log_txt.rfind(f'<!-- {chat_log['log'][-1]['publisher']} -->')
      if chat_log_idx == -1: chat_log_row_idx = chat_log_txt.count('\n')
      else: chat_log_row_idx = chat_log_txt.count('\n', 0, chat_log_idx)
      chat_log_col_idx = len(chat_log_txt.split('\n')[chat_log_row_idx])
      chat_log_pos = (chat_log_row_idx + 1, chat_log_col_idx + 1)
      ###
    else:
      chat_log_txt = ''
      chat_log_pos = (1, 1)

    if user_msg is not None: user_msg_txt = user_msg['content']
    else: user_msg_txt = ''
    assert isinstance(chat_log_txt, str) and isinstance(user_msg_txt, str)
    if not (user_msg_txt := run_editor(
      chat_log=chat_log_txt,
      chat_log_pos=chat_log_pos,
      user_msg=user_msg_txt,
      split=split,
    )): return
    else: yield user_msg_txt