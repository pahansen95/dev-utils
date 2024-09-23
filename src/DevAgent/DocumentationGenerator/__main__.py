"""

CLI for the Documentation Generator

"""
from __future__ import annotations
import logging, os, sys, pathlib, json, yaml
from collections import deque

### DevAgent Imports
from DevAgent.DocumentationGenerator import JSONCapture
from DevAgent.DocumentationGenerator.Task import DocGen
import DevAgent.KnowledgeBase as kb
###

logger = logging.getLogger(__name__)

def gen(
  doc_vision_src: str,
  doc_ref_src: str,
  doc_ctx_src: str,
  corpus_root_src: str | None = None,
  capture_src: str | None = None,
) -> int:
  """Generate Documentation"""
  if capture_src:
    capture = JSONCapture(capture_src)
    capture.write({ 'kind': 'Capture Started' })
  
  if not (corpus_root := pathlib.Path(corpus_root_src)).exists(): raise FileNotFoundError(corpus_root.as_posix())
  if not (doc_vision := pathlib.Path(doc_vision_src)).exists(): raise FileNotFoundError(doc_vision.as_posix())
  if not (_doc_ref := pathlib.Path(doc_ref_src)).exists(): raise FileNotFoundError(_doc_ref.as_posix())
  if not (doc_ctx := pathlib.Path(doc_ctx_src)).exists(): raise FileNotFoundError(doc_ctx.as_posix())

  ### Load the Documentation Ref; TODO Refactor this
  assert _doc_ref.suffix == '.py', _doc_ref.name
  import importlib
  module_spec = importlib.util.spec_from_file_location('doc_ref', _doc_ref)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  doc_ref = module.entrypoint()
  assert isinstance(doc_ref, str) and doc_ref.strip(), doc_ref
  ###

  # Load the Corpus
  try: corpus = kb.Corpus.factory(corpus_root)
  except:
    logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
    return 1

  logger.info('Loading Vision & Context; Generating Documentation')
  sys.stdout.write(
    DocGen.entrypoint(
      doc_vision=doc_vision.read_text(),
      doc_ref=doc_ref,
      extra_doc_ctx=doc_ctx.read_text(),
      corpus=corpus,
      capture=capture,
    )
  )
  capture.write({ 'kind': 'Capture Stopped' })
  capture.close()
  return 0

def render_capture(capture_log_src: str) -> int:
  """Renders the Capture Log to Stdout for Visual Inspection"""
  with open(capture_log_src, 'r') as src:
    for line in src:
      o = json.loads(line)
      sys.stdout.write(f'---\n\n<!-- {o["kind"]} -->\n\n')
      if 'content' in o:
        if isinstance(o['content'], str): sys.stdout.write(o['content'])
        elif isinstance(o['content'], (list, dict)): sys.stdout.write(
          '```YAML\n' \
          + yaml.safe_dump(o['content'], sort_keys=False, default_flow_style=False, default_style='|').strip() \
          + '\n```'
        )
        else: raise NotImplementedError(o['content'])
        sys.stdout.write('\n\n')
  return 0

def main(argv: list[str], env: dict[str, str]) -> int:
  logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'), stream=sys.stderr)

  logger.debug(f'ARGV: {argv}')
  args = deque(a for a in argv if not a.startswith('-'))
  logger.debug(f'Arguments: {args}')
  flags = dict(_parse_flag(f) for f in argv if f.startswith('-'))
  logger.debug(f'Flags: {flags}')

  subcmd = args.popleft()
  try:
    if subcmd.lower() == 'gen': gen(
      doc_vision_src=args.popleft(),
      doc_ref_src=args.popleft(),
      doc_ctx_src=args.popleft(),
      capture_src=flags.get('capture', None),
      corpus_root_src=flags.get('kb', './KnowledgeBase')
    )
    elif subcmd.lower() == 'render-capture': render_capture(
      args.popleft(),
    )
    else: raise RuntimeError(subcmd)
  finally:
    logger.debug('Fin')

### CLI Helpers ###

def _parse_flag(flag: str) -> tuple[str, str]:
  assert flag.startswith('-')
  if '=' in flag: return flag.lstrip('-').split('=', maxsplit=1)
  else: return flag.lstrip('-'), True

def _cleanup():
  logging.shutdown()
  sys.stdout.flush()
  sys.stderr.flush()

if __name__ == '__main__':
  _rc = 2
  try: _rc = main(sys.argv[1:], dict(os.environ))
  except: logger.exception('Unhandled Exception')
  finally: _cleanup()
  exit(_rc)