"""

Import the current set of RFC into the KnowledgeBase

- Fetch the RFC data via rsync
- Filter the set of files to include/exclude (using rsync)
- Generate a md5sum for the set of files
- Add each document to the KB
  - Assemble the Path: Root, Path Prefix, Filename
  - Check if the File's md5 checksum is present in the KB
  - Add to KB if not present

"""
import sys, pathlib
sys.path.insert(0, pathlib.Path(__file__).parent.as_posix())
from _common import *

### Fetch the RFC DATA
subprocess.run(shlex.split(f"""
rsync -a --progress 'rsync.rfc-editor.org::rfcs-text-only' '{RFC_CACHE.as_posix()}'
""".strip()),
  check=True,
  stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)

### Create the PsuedoFS
(RFC_IMPORTS := WORK_CACHE / 'RFC-Imports').mkdir(mode=0o755, exist_ok=True)
rfc_include_globs = [
  'std/std[0-9]*.txt', # All Standards
]
rfc_includes = set(m for p in rfc_include_globs for m in RFC_CACHE.rglob(p))

rfc_exclude_globs = [
  # ...
]
rfc_excludes = set(m for p in rfc_exclude_globs for m in RFC_CACHE.rglob(p))

def link_file(f: pathlib.Path) -> pathlib.Path:
  ln_src = RFC_IMPORTS / f.relative_to(RFC_CACHE)
  if not ln_src.exists():
    ln_src.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    ln_src.symlink_to(f)
  return ln_src

staged_files = list(map(link_file, (f for f in rfc_includes - rfc_excludes if f.is_file())))

### Generate a fingerprint for each file
splits = list(line.split() for line in (
subprocess.run(shlex.split(f"""
md5sum {' '.join(f.relative_to(RFC_IMPORTS).as_posix() for f in staged_files)}
"""), check=True, capture_output=True, cwd=RFC_IMPORTS).stdout.decode().splitlines()
))
assert all(len(s) == 2 for s in splits)
staged_file_fingerprints = { pathlib.Path(file): fingerprint for fingerprint, file in splits }
# logger.debug(f'{staged_file_fingerprints=}')'

### Filter out files w/ duplicate fingerprints
files_to_write = set(
  f for f, fingerprint in staged_file_fingerprints.items()
  if read_fingerprint(f) != fingerprint
)
logger.debug(f'{files_to_write=}')

### Add the file
def add_file(f: pathlib.Path):
  assert not f.is_absolute(), f
  _f = RFC_IMPORTS / f
  _fname = filename(f)
  _fprefix = f'{RFC_KB_PREFIX}/{fileprefix(f)}'
  subprocess.run(shlex.split(f"""
python3 -OO -m DevAgent.KnowledgeBase corpus add \
  --kb={KB_DIR.as_posix()} \
'{_fname}' text 'file:{_f.as_posix()}' '{_fprefix}'
"""), check=True, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
list(map(add_file, files_to_write))

### Cache the fingerprint
def write_fingerprint(f: pathlib.Path):
  (RFC_KB_META / 'fingerprints' / f).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
  (RFC_KB_META / 'fingerprints' / f).write_text(
    staged_file_fingerprints[f]
  )
list(map(write_fingerprint, files_to_write))
