"""

Generate Embeddings for the RFC Imports.

"""
import sys, pathlib
sys.path.insert(0, pathlib.Path(__file__).parent.as_posix())
from _common import *

### Load the set of files from the fingerprints Directory

rfc_docs = [
  f.relative_to(RFC_KB_DIR).parent for f in RFC_KB_DIR.rglob('**/.kind')
  if f.read_text() == 'Document'
]
logger.debug(f'{rfc_docs=}')

def gen_embeds(f: pathlib.Path):
  if len(list((RFC_KB_DIR / f).rglob('semantics/embeds/*.bin'))) > 0: return # Short Circuit
  subprocess.run(shlex.split(f"""
python3 -OO -m DevAgent.KnowledgeBase doc embed \
  --kb={KB_DIR.as_posix()} {f.name} --chunk=1024
"""), check=True, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
list(map(gen_embeds, rfc_docs))
