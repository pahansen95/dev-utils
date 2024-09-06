import os, pathlib, logging, subprocess, shlex
from DevAgent import Ctx

logger = logging.getLogger(__name__)

WORKTREE = pathlib.Path(os.environ['WORK_DIR'])
assert WORKTREE.is_dir()
SRC_DIR = pathlib.Path(os.environ['WORK_SRC'])
assert SRC_DIR.is_dir()
KNOWLEDGE_BASE = WORKTREE / 'meta/KnowledgeBase'

def _load_file_map(root: pathlib.Path, extensions: frozenset[str]) -> dict[pathlib.Path, str]:
  """Load a Mapping of Files to their Contents"""

  name_matches: str = " -o ".join([ f'-name "*.{ext.lstrip('.')}"' for ext in extensions ])
  print_action = r'-printf "%p\0"'

  find_cmd = f"find {root.as_posix()} -type f ( {name_matches} ) {print_action}"

  logger.debug(f"Running: {find_cmd}")
  proc = subprocess.run(shlex.split(find_cmd), capture_output=True, text=True, check=True)
  file_list = sorted([pathlib.Path(p) for p in proc.stdout.split('\0') if p])
  logger.debug(f"Found {len(file_list)} Files...\n{'\n'.join(p.as_posix() for p in file_list)}")
  def _load_file(p: pathlib.Path) -> tuple[pathlib.Path, str]: return p.relative_to(root), p.read_text()
  return dict(map(_load_file, file_list))

def _tree_key(p: pathlib.Path) -> str:
  if p.is_dir(): return p.name + '/'
  else: return p.name

def _assemble_tree_from_file_map(root: pathlib.Path, file_map: dict[pathlib.Path, str]) -> dict[pathlib.Path, dict | str]:
  """Load a Tree from a Mapping of Files to their Contents"""

  tree_paths = sorted(file_map.keys())
  # assert all(k.is_relative_to(root) for k in tree_paths)
  assert not any(k.is_absolute() or k.is_relative_to(root) for k in tree_paths)

  root_key = _tree_key(root)
  root_tree = { root_key: {} }
  for p in tree_paths:
    p_key = _tree_key(p)
    # logger.debug(f'Adding File To Tree: {p.relative_to(root).as_posix()}')
    assert p_key not in root_tree # Ensure No Overwrites
    tree = root_tree[root_key] # Start from root as always
    for parent in p.parents:
      parent_key = _tree_key(parent)
      if parent.as_posix() == '.': continue
      if parent_key not in tree:
        # logger.debug(f'Adding Parent Directory To Tree: {parent.name}')
        tree[parent_key] = {}
      # logger.debug(f'Using Parent SubTree: {parent.name}')
      tree = tree[parent_key]
    # logger.debug(f'Adding Name To SubTree: {p.name}')
    tree[p_key] = file_map[p]
  # logger.debug(f'Final Tree: {root_tree}')
  return root_tree

def spec_factory() -> Ctx.Spec:
  return {
    'about': {
      'Project': Ctx.Spec.About.ProjectStats.factory(WORKTREE),
      'Python': Ctx.Spec.About.PythonStats.factory(WORKTREE),
      'Git': Ctx.Spec.About.GitStats.factory(WORKTREE),
    },
    'src': {
      '/' : {
        **_assemble_tree_from_file_map(SRC_DIR / 'DevAgent', _load_file_map(SRC_DIR / 'DevAgent', frozenset({'.py', '.pyi', '.md'}))),
        "requirements.txt": (SRC_DIR / "requirements.txt").read_text(),
      }
    },
    'kb': _load_file_map(KNOWLEDGE_BASE, frozenset({'.md'})),
  }
