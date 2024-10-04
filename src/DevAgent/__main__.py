"""

An Agent to Assist with development

K.I.S.S: Opt for the dumbest implementation first.

"""
from __future__ import annotations
import logging, pathlib, os, sys, subprocess, hashlib, requests
from typing import Literal
from collections import deque

logger = logging.getLogger(__name__)
is_stdin_tty = os.isatty(sys.stdin.fileno())

class SubCommands:

  @staticmethod
  def chat_tui(
    chat_ctx_src: str,
    chat_log_src: str,
    split: Literal['horizontal', 'vertical'],
  ):
    """Have an interactive Chat with an Agent using a Text User Interface (TUI)
    
    - Assemble the Context & Load or Create the Chat Log
    - LOOP: Chat w/ the Agent
      - Prompt the User for a Message
        - If user provides no input, break
      - Prompt the LLM for a Response
      - Sync the ChatLog to Disk
    - Commit changes to the chat log
    """
    if not is_stdin_tty: raise RuntimeError('Interactive Chat is only supported in TTY Mode')

    from . import tui, AgentCtx
    from .Utils import Chat, Git
    from .Utils.NatLang import load_chat_interface
    from .Utils.NatLang.chat import MODEL_BEHAVIOR, Message

    chat = load_chat_interface(**os.environ)

    GIT_WORKTREE = pathlib.Path(os.environ.get('WORK_DIR'))
    
    if not (chat_ctx_file := pathlib.Path(chat_ctx_src)).exists(): raise RuntimeError(f'File Not Found: {chat_ctx_src}')
    _load_chat_ctx = AgentCtx.load_spec_factory(chat_ctx_file)
    logger.debug(f'Example Chat Context...\n{AgentCtx.CtxUtils.render_ctx(_load_chat_ctx())}')

    if not (chat_log_file := pathlib.Path(chat_log_src)).exists() or chat_log_file.stat().st_size <= 0: chat_log = Chat.Log.factory()
    else: chat_log = Chat.Log.unmarshal(chat_log_file.read_bytes())
    chat_hash = hashlib.md5(Chat.Log.marshal(chat_log)).digest()

    def _get_chat_log() -> Chat.Log | None: return chat_log
    def _get_user_msg() -> Chat.Message | None:
      if not chat_log['log'] or not chat_log['log'][-1]['publisher'].startswith('user:'): return None
      else: return chat_log['log'].pop()
    def _sync_chat_log(): chat_log_file.write_bytes(Chat.Log.marshal(chat_log))
    def _commit_chat_log():
      try: Git.sync(
        msg='Sync Chat Log',
        pathspecs=[ chat_log_file.relative_to(GIT_WORKTREE).as_posix() ],
        worktree=GIT_WORKTREE,
      )
      except subprocess.CalledProcessError as e: logger.warning(f'Failed to Commit the Chat Log; Please manually sync:\n\n{((e.stdout or "").rstrip() + "\n\n" + (e.stderr or "").lstrip()).strip()}')
      except Exception as e: logger.exception('Failed to Commit the Chat Log; Please manually sync.')
    def _chat_msg_to_llm_msg(msg: Chat.Message) -> Message:
      role = msg['publisher'].split(':', maxsplit=1)[0]
      if role == 'agent': role = 'assistant'
      elif role == 'user': role = 'user'
      else: raise ValueError(f'Unknown Role: {role}')
      return { 'role': role, 'content': msg['content'] }
    def _llm_chat() -> Chat.Message: return Chat.Message.factory(
      publisher=f'agent:llm:{chat.provider.chat_model_identifier}',
      content=chat.chat(
        { 'role': 'user', 'content': 'I will first provide you Contextual Information, then specify your role & behaviors and then provide the conversation you are participating in.' },
        { 'role': 'assistant', 'content': 'I understand, please provide the context.' },
        { 'role': 'user', 'content': AgentCtx.CtxUtils.render_ctx(_load_chat_ctx()) },
        { 'role': 'assistant', 'content': 'Please provide the role & behaviors you expect of me.' }, # Maintain the User/Assistant turn taking
        { 'role': 'user', 'content': MODEL_BEHAVIOR },
        { 'role': 'assistant', 'content': 'Please provide the conversation I am a participant in: I will fulfill my role & embody the desired behaviors; my responses will be informed by the provided context.' },
        *map(_chat_msg_to_llm_msg, chat_log['log'])
      ),
    )

    logger.info('Starting Conversation...')
    for user_resp in tui.editor_loop(
      chat_log_factory=_get_chat_log,
      user_msg_factory=_get_user_msg,
      split=split,
    ):
      assert user_resp.strip() # The Editor Loop will return when the user provides no input
      chat_log['log'].append(Chat.Message.factory(publisher='user:tty', content=user_resp))
      logger.debug('Syncing User Message to Chat Log...')
      _sync_chat_log()
      logger.info(f'Waiting for Response from Model `{chat.provider.chat_model_identifier}`...')
      try: chat_log['log'].append(_llm_chat())
      except: raise # TODO: Handle Errors
      logger.debug('Syncing Model Response to Chat Log...')
      _sync_chat_log()
    
    logger.info('Conversation has concluded')

    diff = hashlib.md5(Chat.Log.marshal(chat_log)).digest() != chat_hash

    logger.debug('Committing Chat Log to Git Worktree...')
    if diff and chat_log_file.is_relative_to(GIT_WORKTREE): _commit_chat_log()
    elif diff: logger.warning(f'Skip Committing Chat Log: Chat Log is not in the Git Worktree: {chat_log_file.as_posix()}')
    else: logger.debug('No Changes to Chat Log')

def _parse_flag(flag: str) -> tuple[str, str]:
  assert flag.startswith('-')
  if '=' in flag: return flag[1:].split('=', maxsplit=1)
  else: return flag[1:], True

def main(argv: list[str], env: dict[str, str]) -> int:
  logger.debug(f'ARGV: {argv}')
  args = deque(a for a in argv if not a.startswith('-'))
  logger.debug(f'Arguments: {args}')
  flags = dict(_parse_flag(f) for f in argv if f.startswith('-'))
  logger.debug(f'Flags: {flags}')
  subcmd = args.popleft()
  if subcmd.lower() == 'chat': return SubCommands.chat_tui(
    chat_ctx_src=args.popleft(),
    chat_log_src=args.popleft(),
    split=flags.get('split', 'horizontal').lower(),
  )
  else: raise RuntimeError(f'Unknown Subcommand: {subcmd}')

if __name__ == '__main__':
  logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'), stream=sys.stderr)
  _rc = 2
  try:
    _rc = main(sys.argv[1:], os.environ)
  except SystemExit as e:
    _rc = e.code
  except RuntimeError as e:
    logger.critical(f'Error Encountered: {e}')
  except Exception as e:
    logger.exception(e)
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    exit(_rc)
