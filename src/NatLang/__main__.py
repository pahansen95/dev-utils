"""

The Package Entrypoint

"""

import logging, os, sys, contextlib, pathlib, io, json
from typing import TextIO, BinaryIO
from collections import deque

from NatLang import protocols as p

SCRIPT = pathlib.Path(__file__)
CONTEXT = SCRIPT.parent # The context of Script
logger = logging.getLogger(__package__ if __name__ == '__main__' else __name__)

def chat(
  src: BinaryIO,
  sink: BinaryIO,
):
  """Simple CLI Interface to Chat with a LLM; reads a chat log or single message from stdin & writes the updated (or new) chat log to stdout"""

  ### TODO: Load the Model Provider & Config

  model_name: str = ...
  provider: p.ModelProvider = ... # TODO: Inject ProviderSession
  llm: p.Model = provider.models[model_name]
  if llm.chat is None: raise RuntimeError(f'Model {llm.name} does not support Chat')
  chat = llm.chat
  custom_properties: p.Properties | None = None # TODO: Load custom properties
  if custom_properties is None: custom_properties = llm.props # Use default properties
  model_tuner: p.ModelTuner = ...
  marshal: p.SeDer.Marshal[p.Conversation] = ...
  unmarshal: p.SeDer.Unmarshal[p.Conversation] = ...

  # Load the Conversation
  content_stream = src.read()
  try: # First attempt ot load 
    convo: p.Conversation = unmarshal(content_stream)
  except: # Otherwise treat it like a single message
    convo: p.Conversation = ...
    content: p.CONTENT = content_stream
    chat_msg: p.Message = {
      'role': 'user',
      'content': content,
    }
    convo.add_nodes(chat_msg)

  chat_logs = convo.chat_logs()
  # Get the first available chat log
  try: chat_log = next(chat_logs)
  except StopIteration: raise RuntimeError('Conversation is empty')
  # TODO: Support multi-threaded conversations
  try: next(chat_logs)
  except StopIteration: pass
  else: raise NotImplementedError(f'Multi-Threaded conversations not currently supported.')
  logger.info(f'Last message in the chat is from a `{chat_log[-1].role}`')

  ### Prompt the LLM
  _log_prefix = llm.parse_instructions( { 'role': 'system', 'content': ... } )   
  with model_tuner as _model:
    assert _model is llm and _model.chat is not None and _model.chat is chat
    assert llm.props is custom_properties
    resp = chat( *( _log_prefix + chat_log ) )
    
  assert resp.role == 'assistant'
  convo.add_nodes(resp)
  convo.add_edges({ 'u': chat_log[-1], 'v': resp, 'k': 'chat' })

  ### Write back results
  sink.write( marshal(convo) )

def main(
  args: deque[str],
  kwargs: dict[str, str],
  remainder: deque[str],
  env: dict[str, str],
  stdout: TextIO,
) -> bool:

  class E(Exception): ...

  def _pop_arg(name: str) -> str:
    try: return args.popleft()
    except IndexError: raise E(f'missing positional arg: `{name.upper()}`')
  NO_DEFAULT = type('NO_DEFAULT', (), {})
  def _get_kwarg(k: str, default: str | bool | type[NO_DEFAULT] = NO_DEFAULT) -> str:
    assert default is NO_DEFAULT or isinstance(default, (str, bool))
    try: return kwargs.get(k, default) if default is not NO_DEFAULT else kwargs[k]
    except KeyError: raise E(f'Missing Expected Flag: `--{k}`')

  try:

    subcmd = _pop_arg('subcmd')

    if subcmd == 'chat':

      try: ... # TODO
      except Exception as e: raise E('Chat Failed') from e

    elif subcmd == 'embed':

      try: raise NotImplementedError
      except Exception as e: raise E('Embedding Failed') from e

    else: raise E(f'Unknown Subcommand: {subcmd}')

  except E as e:
    logger.critical(str(e))
    logger.info(str(e), exc_info=True)
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

    args = deque(a for a in argv if not a.startswith('-'))
    logger.debug(f'{args=}')
    flags = dict(CLI.parse_flag(f) for f in argv if f.startswith('-'))
    logger.debug(f'{flags=}')
    return (args, flags, deque(remainder))

if __name__ == "__main__":
  RC = 2
  with CLI.session() as _ok:
    if _ok: RC = 0 if main(
      *CLI.parse_argv(sys.argv[1:]),
      dict(os.environ),
      sys.stdout,
    ) else 1
  exit(RC)
  