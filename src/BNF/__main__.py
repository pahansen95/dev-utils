"""

The Package Entrypoint

"""

import logging, os, sys, contextlib, pathlib, json, io
from typing import *
from collections.abc import *
from types import *
from collections import deque
from dataclasses import dataclass, field, KW_ONLY
from NetCfg.Lex import BNF

SCRIPT = pathlib.Path(__file__)
CONTEXT = SCRIPT.parent # The context of Script
logger = logging.getLogger(__package__ if __name__ == '__main__' else __name__)

def subcmd_tokenize(
  src: TextIO,
  sink: TextIO,
):
  """tokenizes the input stream into a jsonl file where each line is a token"""
  logger.info('Tokenizing Grammar')
  grammar = src.read()
  tokenizer = BNF.Lexer()
  for token in tokenizer(grammar):
    sink.write(json.dumps({
      'kind': token.kind.name,
      'start': token.start,
      'stop': token.stop,
      'lineno': token.lineno,
      'colno': token.colno,
      'val': token.value,
    }) + '\n')

def subcmd_parse(
  src: TextIO,
  sink: TextIO
):
  """Convert a previously parsed jsonl file of tokens into a parse tree"""
  tokens: list[BNF.Token] = []
  for line in src.readlines():
    if not line.strip(): continue
    raw_token = json.loads(line)
    tokens.append(BNF.Token(
      BNF.TERMINAL_LOOKUP[raw_token['kind']],
      raw_token['val'],
      raw_token['start'],
      raw_token['stop'],
      raw_token['lineno'],
      raw_token['colno']
    ))
  parse_tree = BNF.Parser(tokens)()
  logger.debug(f'{parse_tree}')
  sink.write(str(parse_tree))

@contextlib.contextmanager
def load_stream(stream: str, default: TextIO = None) -> Generator[TextIO, None, None]:
  if stream == '-':
    if default is None: raise ValueError
    yield default
  else:
    with open(stream, 'r') as s:
      yield s

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

    if subcmd == 'tokenize':
      with (
        load_stream(_pop_arg('input', '-'), stdin) as src,
        load_stream(_pop_arg('output', '-'), stdout) as sink,
      ):
        try: subcmd_tokenize(src, sink)
        except Exception as e: raise E(f'Failed to {subcmd}') from e
    elif subcmd == 'parse':
      with (
        load_stream(_pop_arg('input', '-'), stdin) as src,
        load_stream(_pop_arg('output', '-'), stdout) as sink,
      ):
        try: subcmd_parse(src, sink)
        except Exception as e: raise E(f'Failed to {subcmd}') from e
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
  