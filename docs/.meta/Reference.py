"""

The Documentation Reference Material.

This is a placeholder while we are actively developing things.

"""
import os, pathlib, logging, subprocess, shlex
from DevAgent.AgentCtx.CtxUtils import load_filemap, render_section

logger = logging.getLogger(__name__)

WORKTREE = pathlib.Path(os.environ['WORK_DIR'])
assert WORKTREE.is_dir()
SRC_DIR = pathlib.Path(os.environ['WORK_SRC'])
assert SRC_DIR.is_dir()

def entrypoint() -> str:
  return render_section(
    heading='#',
    title='DevAgent Source Code',
    content=None,
    files=load_filemap(SRC_DIR / 'DevAgent', frozenset({'*.py', '*.pyi'}))
  )
