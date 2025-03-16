"""

The Package Entrypoint

"""

import logging, os, sys, contextlib, pathlib, io, json, time
from typing import TextIO, BinaryIO
from collections import deque

import NatLang as nl

SCRIPT = pathlib.Path(__file__)
CONTEXT = SCRIPT.parent # The context of Script
logger = logging.getLogger(__package__ if __name__ == '__main__' else __name__)

def subcmd_prompt(
  model_slug: str,
  src: str | BinaryIO,
  sink: BinaryIO,
  env: dict[str, str],
  provider_cfg: dict | None,
  model_tuning: dict | None,
):
  """Simple CLI Interface to prompt a LLM; takes prompt text & writes the reply to stdout"""

  provider_name, model_name = model_slug.split(':')
  provider = nl.load_provider_from_env(provider_name, env, provider_cfg or {})
  if not provider.supports(model_name, 'chat'): raise RuntimeError(f'Model {model_slug} does not support Chat')
  assert provider.chat is not None
  chat = provider.chat

  # Load the Message
  if not isinstance(src, str):
    assert hasattr(src, 'read')
    content = src.read().decode()
  else: content = src

  ### Prompt the LLM
  with provider.tune(model_name, **( model_tuning or {} )):
    resp = chat( model_name, *(
      { 'role': nl.chat.Role.PLATFORM, 'content': 'follow all instructions provided' },
      { 'role': nl.chat.Role.USER, 'content': content }
    ) )
    
  assert resp['role'] == nl.chat.Role.AGENT

  ### Write back results
  content: str = resp['content']
  assert isinstance(content, str)
  sink.write( content.encode() )

def subcmd_chat(
  model_slug: str,
  src: str | BinaryIO,
  sink: BinaryIO,
  env: dict[str, str],
  provider_cfg: dict | None,
  model_tuning: dict | None,
):
  """Simple CLI Interface to Chat with a LLM; reads a chat log or single message from stdin & writes the updated (or new) chat log to stdout"""

  ### TODO: Load the Model Provider & Config

  provider_name, model_name = model_slug.split(':')
  provider = nl.load_provider_from_env(provider_name, env, provider_cfg)
  if not provider.supports(model_name, 'chat'): raise RuntimeError(f'Model {model_slug} does not support Chat')
  assert provider.chat is not None
  chat = provider.chat
  marshal_convo = nl.chat.Conversation.marshal
  unmarshal_convo = nl.chat.Conversation.unmarshal

  # Load the Message
  if not isinstance(src, str):
    assert hasattr(src, 'read')
    content_stream = src.read().decode()
  else: content_stream = src
  try: # Attempt to unmarshal a Conversation
    convo = unmarshal_convo(content_stream)
  except: # Otherwise treat it like raw input & create a new conversation
    logger.debug('Failed to unmarshal Chat Conversation', exc_info=True)
    convo = nl.chat.Conversation()
    chat_msg: nl.chat.ChatMessage = { 'role': nl.chat.Role.USER, 'content': content_stream, 'props': {
      'created_at': time.time_ns(),
    } }
    convo.new_chat(chat_msg)

  chat_logs = convo.chat_logs()
  # Get the first available chat log
  try: chat_log = next(chat_logs)
  except StopIteration: raise RuntimeError('Conversation is empty')
  # TODO: Support multi-threaded conversations
  try: next(chat_logs)
  except StopIteration: pass
  else: raise NotImplementedError(f'Multi-Threaded conversations not currently supported.')
  logger.info(f'Last message in the chat is from a `{convo[chat_log[-1]]["role"]}`')

  ### Prompt the LLM
  instructions: list[nl.chat.ChatMessage] = [
    { 'role': nl.chat.Role.PLATFORM, 'content': 'follow all instructions provided' },
  ]
  with provider.tune(model_name, **( model_tuning or {} )):
    resp = chat( model_name, *( instructions + [ convo[mid] for mid in chat_log ] ) )
    
  assert resp['role'] == nl.chat.Role.AGENT
  convo.add_reply(resp, chat_log[-1])

  ### Write back results
  sink.write( marshal_convo(convo) )

def main(
  args: deque[str],
  kwargs: dict[str, str],
  remainder: deque[str],
  env: dict[str, str],
  stdin: TextIO,
  stdout: TextIO,
) -> bool:

  class E(Exception): ...
  NO_DEFAULT = type('NO_DEFAULT', (), {})
  def _pop_arg(name: str, default: str | None = NO_DEFAULT) -> str:
    try: return args.popleft()
    except IndexError:
      if default is NO_DEFAULT: raise E(f'missing positional arg: `{name.upper()}`')
      return default
  def _get_kwarg(k: str, default: str | bool | None | type[NO_DEFAULT] = NO_DEFAULT) -> str:
    assert default is NO_DEFAULT or default is None or isinstance(default, (str, bool))
    try: return kwargs.get(k, default) if default is not NO_DEFAULT else kwargs[k]
    except KeyError: raise E(f'Missing Expected Flag: `--{k}`')

  try:

    subcmd = _pop_arg('subcmd')

    if subcmd in { 'prompt', 'chat' }:

      src = _pop_arg(subcmd, '-')
      if src == '-': src = stdin.buffer

      model_tuning = _get_kwarg('tune', None)
      if model_tuning is not None: model_tuning = json.loads(model_tuning)

      provider_cfg = _get_kwarg('provider', None)
      if provider_cfg is not None: provider_cfg = json.loads(provider_cfg)

      if subcmd == 'prompt': subcmd_fn = subcmd_prompt
      elif subcmd == 'chat': subcmd_fn = subcmd_chat
      else: raise NotImplementedError(subcmd)

      try: subcmd_fn(
        model_slug=_get_kwarg('model', 'openai:gpt-4o-mini'),
        src=src, sink=stdout.buffer,
        env=env,
        provider_cfg=provider_cfg,
        model_tuning=model_tuning,
      )
      except Exception as e: raise E(f'{subcmd} Failed') from e

    elif subcmd == 'embed':

      try: raise NotImplementedError
      except Exception as e: raise E('Embedding Failed') from e

    else: raise E(f'Unknown Subcommand: {subcmd}')

  except E as e:
    logger.info(str(e), exc_info=True)
    logger.critical(str(e))
    return False
  return True

class CLI:

  @classmethod
  @contextlib.contextmanager
  def session(cls):
    try:
      try:
        logging.basicConfig(stream=sys.stderr, level=os.environ.get('LOG_LEVEL', 'INFO'))
      except Exception as e:
        logging.basicConfig(stream=sys.stderr, level='INFO')
        logger.critical(f'Bad Log Configuration: {e}')
        yield False
      else:
        logger.debug('inizio')
        yield True # Any CLI Exceptions will be raised here
    except:
      logger.critical('Unhandled Exception', exc_info=True)
    finally:
      logger.debug('fin')
      logging.shutdown()
      sys.stdout.flush()
      sys.stderr.flush()
  
  @classmethod
  def parse_flag(cls, flag: str) -> tuple[str, str]:
    assert flag.startswith('-')
    if '=' in flag: return flag.lstrip('-').split('=', maxsplit=1)
    else: return flag.lstrip('-'), True

  @classmethod
  def parse_argv(cls, argv: list[str]) -> tuple[deque[str], dict[str, str], deque[str]]:
    """Parses Argv returning ( args, kwargs, remainder )"""
    remainder = []
    if '--' in argv:
      idx = argv.index('--')
      remainder = argv[idx+1:]
      argv = argv[:idx]
    logger.debug(f'{remainder=}')

    args = deque(a for a in argv if not (a.startswith('-') and a != '-'))
    logger.debug(f'{args=}')
    flags = dict(CLI.parse_flag(f) for f in argv if (f.startswith('-') and f != '-'))
    logger.debug(f'{flags=}')
    return (args, flags, deque(remainder))

if __name__ == "__main__":
  RC = 2
  with CLI.session() as _ok:
    if _ok: RC = 0 if main(
      *CLI.parse_argv(sys.argv[1:]),
      dict(os.environ),
      sys.stdin,
      sys.stdout,
    ) else 1
  exit(RC)
  