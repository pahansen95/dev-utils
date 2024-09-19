"""

Context Utilities

"""
from __future__ import annotations
import subprocess, pathlib, logging, shlex
from . import FileMap_t, Spec

logger = logging.getLogger(__name__)

def load_filemap(root: pathlib.Path, matches: frozenset[str]) -> dict[pathlib.Path, str]:
  """Load a Mapping of Files to their Contents; Path's are relative to the provided root"""

  name_matches: str = " -o ".join([ f'-name "{match}"' for match in matches])
  print_action = r'-printf "%p\0"'

  find_cmd = f"find {root.as_posix()} -type f ( {name_matches} ) {print_action}"

  logger.debug(f"Running: {find_cmd}")
  proc = subprocess.run(shlex.split(find_cmd), capture_output=True, text=True, check=True)
  file_list = sorted([pathlib.Path(p) for p in proc.stdout.split('\0') if p])
  logger.debug(f"Found {len(file_list)} Files...\n{'\n'.join(p.as_posix() for p in file_list)}")
  def _load_file(p: pathlib.Path) -> tuple[pathlib.Path, str]: return p.relative_to(root), p.read_text()
  return dict(map(_load_file, file_list))

def load_project_stats(worktree: pathlib.Path) -> Spec.About.ProjectStats:
  return {
    'README': (worktree / 'README.md').read_text(),
  }

def load_python_stats() -> Spec.About.PythonStats:
  return {
    'Version': subprocess.run(['python', '--version'], check=True, capture_output=True).stdout.decode().strip(),
  }

def load_git_stats(worktree: pathlib.Path) -> Spec.About.PythonStats:
  return {
    'Version': subprocess.run(['git', '--version'], check=True, capture_output=True).stdout.decode().strip(),
    'Branches': subprocess.run(['git', 'branch', '--list'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
    'Log': 'Last 3 Commits:\n' + subprocess.run(['git', 'log', '--oneline', '-n', '3'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
    'HEAD': subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
  }

FILE_EXT_LOOKUP: dict[str, str] = {
  **dict((k, 'PlainText') for k in ('.txt', '.text')),
  **dict((k, 'Markdown') for k in ('.md', '.markdown')),
  **dict((k, 'JSON') for k in ('.json', '.jsonl')),
  **dict((k, 'YAML') for k in ('.yml', '.yaml')),
  **dict((k, 'TOML') for k in ('.tml', '.toml')),
  **dict((k, 'Python') for k in ('.py', '.pyi')),
  **dict((k, 'Shell') for k in ('.sh', '.bash', '.zsh')),
  **dict((k, 'PowerShell') for k in ('.ps1')),
}
def render_file(path: pathlib.Path | str, content: str, blurb: str | None = None, heading: str = '####') -> str:
  if isinstance(path, str): path = pathlib.Path(path)
  assert isinstance(path, pathlib.Path)
  fragment = f"{heading} {path.as_posix()}\n\n"
  if blurb: fragment += f'{blurb}\n\n'
  codekind = FILE_EXT_LOOKUP.get(path.suffix, '') # If we can't find it don't specify anything
  if codekind == 'Markdown': codeblock = '````'
  else: codeblock = '```'
  fragment += f'''
{codeblock}{codekind}
{content}
{codeblock}
'''.strip()
  return fragment

def render_filemap(filemap: FileMap_t, heading: str = '####') -> str:
  return '\n\n'.join(
    render_file(p, c, heading=heading)
    for p, c in filemap.items()
  )

def render_plainmap(plainmap: dict[str, str], heading: str = '####') -> str:
  return '\n\n'.join([f'{heading} {key}\n\n```plaintext\n{value}\n```'.strip() for key, value in plainmap.items()]).strip()

def render_ctx_about(about: Spec.About) -> str:
  return f"""
## About

### README

````markdown
{about['Project']['README'].strip()}
````

### Python Stats

{render_plainmap(about['Python']).strip()}

### Git Stats

{render_plainmap(about['Git']).strip()}
""".strip()

def render_section(heading: str, title: str, content: str | None, files: FileMap_t | None) -> str:
  assert all(c == '#' for c in heading)
  fragment = f'''{heading} {title}\n\n'''
  if content: fragment += content.strip() + '\n\n'
  if files: fragment += render_filemap(files, heading='#'+heading).strip()
  return fragment

def render_ctx(ctx: Spec) -> str:
  """Renders the Context into a Human Readable Document"""

  doc = "# Context"
  assert 'about' in ctx
  doc += '\n\n' + render_ctx_about(ctx['about'])
  if 'src' in ctx: doc += '\n\n' + render_section('##', 'Source Code', None, ctx['src'])
  if 'docs' in ctx: doc += '\n\n' + render_section('##', 'Documentation', None, ctx['docs'])
  for entry in ctx.get('other', []):
    doc += '\n\n' + render_section('##', entry['title'], entry.get('content'), entry.get('files'))

  return doc.strip()

### TODO: Refactor
# def render_tree(tree: dict[str, dict | str]) -> str:
#   """Render a Directory Tree into a Deeply Nested Markdown List"""
#   o = ""
#   class _Node(TypedDict):
#     depth: int
#     prefix: str
#     tree: dict[str, dict | str]
#   stack: deque[_Node] = deque([{ 'depth': 0, 'prefix': '', 'tree': tree }])
#   htab = lambda depth: '  ' * depth
#   while stack:
#     node = stack.popleft()
#     if (node['depth'] - 1) >= 0: o += f"{htab(node['depth'] - 1)}- `{node['prefix'].rstrip('/')}/`\n"
#     for name, child in sorted(node['tree'].items(), key=lambda x: x[0]):
#       assert isinstance(child, (dict, str)), type(child)
#       # child_path = node['prefix'].rstrip('/') + '/' + name
#       if isinstance(child, str): o += f"{htab(node['depth'])}- `{name}`\n"
#       else: stack.append({ 'depth': node['depth'] + 1, 'prefix': name, 'tree': child })
#   return o.strip()
