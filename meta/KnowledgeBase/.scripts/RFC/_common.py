from __future__ import annotations
import subprocess, shlex, os, sys, pathlib, logging

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=os.environ.get('LOG_LEVEL', 'INFO'))

WORK_CACHE = pathlib.Path(os.environ['WORK_CACHE'])
assert WORK_CACHE.exists()
WORK_DIR = pathlib.Path(os.environ['WORK_DIR'])
assert WORK_DIR.exists()
(RFC_CACHE := WORK_CACHE / 'RFC').mkdir(mode=0o755, exist_ok=True)
# KB_DIR = pathlib.Path(WORK_DIR / 'meta/KnowledgeBase')
KB_DIR = pathlib.Path(WORK_CACHE / 'TestKnowledgeBase')
assert KB_DIR.exists()
RFC_KB_PREFIX = 'RFC'
(RFC_KB_DIR := KB_DIR / RFC_KB_PREFIX).mkdir(mode=0o755, parents=False, exist_ok=True)
(RFC_KB_META := RFC_KB_DIR / '.meta').mkdir(mode=0o755, parents=False, exist_ok=True)

def read_fingerprint(f: pathlib.Path) -> str:
  assert not f.is_absolute(), f
  if not (_f := (RFC_KB_META / 'fingerprints' / f)).exists(): return 'True'
  else: return _f.read_text()
def fileprefix(f: pathlib.Path) -> str:
  assert not f.is_absolute(), f
  return f.parent.as_posix()
def filename(f: pathlib.Path) -> str:
  assert not f.is_absolute(), f
  return f'Rfc{f.name.capitalize().rsplit('.', maxsplit=1)[0]}'
