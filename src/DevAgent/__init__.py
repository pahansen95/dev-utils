from __future__ import annotations
from typing import TypedDict
from collections.abc import Generator, Iterable
import time, subprocess, pathlib, importlib.util, json, re, logging
from collections import deque

logger = logging.getLogger(__name__)

class Chat:
  """Semantic Type for a Chat Conversation"""

  class Message(TypedDict):
    publisher: str
    """Who authored the message"""
    content: str
    """The message content"""
    created: int
    """When the message was created as a Unix Nanosecond Timestamp"""

    @staticmethod
    def factory(
      publisher: str,
      content: str,
      created: int = None,
    ) -> Chat.Message:
      return {
        'publisher': publisher,
        'content': content,
        'created': created or time.time_ns(),
      }

  class Log(TypedDict):
    log: list[Chat.Message]

    @staticmethod
    def factory(
      msgs: Iterable[Chat.Message] = []
    ) -> Chat.Log:
      return { 'log': list(msgs) }

    @staticmethod
    def marshal(log: Chat.Log) -> bytes:
      """
      
      Marshal the Chat Log into a Human Readable Format:

      ````markdown
      <!-- LOG METADATA [JSON Obj] -->

      ---

      <!-- MSG METADATA [JSON Obj] -->

      [Message Content]

      ---
      ````

      The trailing `---` is stripped.
      
      """

      def _embed(obj: dict, kind: str) -> str: return f'<!-- {kind.upper()} METADATA {json.dumps(obj)} -->'

      return (
        '\n'.join([
          _embed({
            'size': len(log['log']),
          }, 'log'),
          *[
            '\n'.join([
              '\n---\n',
              _embed({k: v for k, v in msg.items() if k in {'publisher', 'created'}}, 'msg') + '\n',
              msg['content'].strip(),
            ]) for idx, msg in enumerate(log['log'])
          ],
        ]).strip() + '\n'
      ).encode()

    LOG_METADATA_RE = re.compile(r'<!-- LOG METADATA (.+) -->')
    MSG_METADATA_RE = re.compile(r'---\s+<!-- MSG METADATA (.+) -->')

    @staticmethod
    def unmarshal(data: bytes) -> 'Chat.Log':
      """
      Unmarshal the Chat Log from a Human Readable Format

      Reverses the marshal process, reconstructing the Chat.Log object
      from the given byte string.
      """
      text = data.decode().strip()

      # Extract metadata from the first line
      log_metadata_match = Chat.Log.LOG_METADATA_RE.match(text)
      if not log_metadata_match: raise ValueError('Missing Log Metadata')
      log_metadata = json.loads(log_metadata_match.group(1))
      log_size = log_metadata['size']
      text = text[log_metadata_match.end():]

      msg_metadata_matches = list(Chat.Log.MSG_METADATA_RE.finditer(text))
      if len(msg_metadata_matches) != log_size: raise ValueError(f'Mismatched Log Size: Got {len(msg_metadata_matches)}, Expected {log_size}')

      messages = []      
      for idx, match in enumerate(msg_metadata_matches):
        msg_metadata = json.loads(match.group(1))
        msg_start = match.end()
        if idx + 1 < len(msg_metadata_matches): msg_end = msg_metadata_matches[idx + 1].start()
        else: msg_end = len(text)
        msg_content = text[msg_start:msg_end].strip()
        messages.append({
          'publisher': msg_metadata['publisher'],
          'content': msg_content,
          'created': msg_metadata['created'],
        })

      return { 'log': messages }

class Git:
  """Semantic Type for basic Git Utility"""
  
  @staticmethod
  def sync(
    msg: str,
    pathspecs: list[str],
    worktree: pathlib.Path,
  ):
    subprocess.run(['git', 'add', *pathspecs], check=True, cwd=worktree.as_posix(), capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', msg, *pathspecs], check=True, cwd=worktree.as_posix(), capture_output=True, text=True)

# class Ctx:
#   """Semantic Type for a Project Context

#   Project Context is used to provide some Relevant Information about the Project to the DevAgent.

#   A Context consists of:
#     - Project README
#     - Project Source...
#       - The Directory Tree
#       - Individual Files whose content will be injected
#     - Project Documentation... TODO
#     - Knowledge Base... TODO
#   """

#   @staticmethod
#   def load(filepath: pathlib.Path) -> Ctx.Spec:
#     """Load a Project Context Module from a File & then call the `spec_factorty` function to get the Context Specification"""
#     module_spec = importlib.util.spec_from_file_location('spec_module', filepath)
#     module = importlib.util.module_from_spec(module_spec)
#     module_spec.loader.exec_module(module)
#     return module.spec_factory()

#   @staticmethod
#   def render(spec: Ctx.Spec) -> str:
#     """Render a Project Context Specification into a Human Readable Document"""

#     def _render_tree(tree: dict[str, dict | str]) -> str:
#       """Render a Directory Tree into a Deeply Nested Markdown List"""
#       o = ""
#       class _Node(TypedDict):
#         depth: int
#         prefix: str
#         tree: dict[str, dict | str]
#       stack: deque[_Node] = deque([{ 'depth': 0, 'prefix': '', 'tree': tree }])
#       htab = lambda depth: '  ' * depth
#       while stack:
#         node = stack.popleft()
#         if (node['depth'] - 1) >= 0: o += f"{htab(node['depth'] - 1)}- `{node['prefix'].rstrip('/')}/`\n"
#         for name, child in sorted(node['tree'].items(), key=lambda x: x[0]):
#           assert isinstance(child, (dict, str)), type(child)
#           # child_path = node['prefix'].rstrip('/') + '/' + name
#           if isinstance(child, str): o += f"{htab(node['depth'])}- `{name}`\n"
#           else: stack.append({ 'depth': node['depth'] + 1, 'prefix': name, 'tree': child })
#       return o.strip()
    
#     def _render_file(filepath: str, content: str) -> str:
#       codeblock = '```'
#       if filepath.endswith(('.md', '.markdown')):
#         codeblock = '````'
#         codekind = 'Markdown'
#       elif filepath.endswith(('.py', '.pyi', '.pyc')): codekind = 'Python'
#       elif filepath.endswith(('.ebnf', '.bnf')): codekind = 'EBNF'
#       elif filepath.endswith(('.x', '.langx')): codekind = 'LangX'
#       elif filepath.endswith(('.json', '.jsonl')): codekind = 'JSON'
#       elif filepath.endswith(('.yaml', '.yml')): codekind = 'YAML'
#       elif filepath.endswith(('.sh', '.bash', '.zsh')): codekind = 'Shell'
#       elif filepath.endswith(('.ps1',)): codekind = 'PowerShell'
#       else: codekind = 'Plaintext'
#       return f"""
# ### {filepath}

# {codeblock}{codekind}
# {content}
# {codeblock}
# """.strip()
    
#     def _render_src_files(tree: dict[str, dict | str]) -> Generator[str, None, None]:
#       """Sequentially Render the Source Files"""
#       class _Node(TypedDict):
#         prefix: str
#         tree: dict[str, dict | str]
#       stack: deque[_Node] = deque([{ 'prefix': '/', 'tree': tree }])
#       while stack:
#         node = stack.popleft()
#         for name, child in node['tree'].items():
#           assert isinstance(child, (dict, str)), type(child)
#           child_path = node['prefix'].rstrip('/') + '/' + name
#           if isinstance(child, str):
#             if child: yield _render_file(child_path, child)
#           else: stack.append({ 'prefix': child_path, 'tree': child })

#     def _render_file_map(file_map: dict[pathlib.Path, str]) -> str:
#       """Sequentially Render the Files in the File Map"""
#       return '\n\n'.join([_render_file(p.as_posix(), c) for p, c in file_map.items()])

#     def _render_stats(stats: dict[str, str]) -> str:
#       return '\n'.join([f'### {stat}\n\n```plaintext\n{value}\n```\n' for stat, value in stats.items()]).strip()

#     return f"""
# # Context

# ## About the Project

# ### README

# `````markdown
# {spec['about']['Project']['README'].strip()}
# `````

# ## Python Stats

# {_render_stats(spec['about']['Python']).strip()}

# ## Git Stats

# {_render_stats(spec['about']['Git']).strip()}

# ## Knowledge Base

# {_render_file_map(spec['kb']).strip()}

# ## Project Source

# ### Source Tree

# {_render_tree(spec['src']).strip()}

# ### Source Files

# {'\n\n'.join(_render_src_files(spec['src'])).strip()}

# """.strip() + '\n' # Markdown Spec says to end with a newline

#   class Spec(TypedDict):
#     about: Ctx.Spec.About
#     """The Project Metadata"""
#     src: dict[str, dict | str]
#     """The Project Source organized as a Directory Tree"""
#     kb: dict[str, str]
#     """Project Knowledge Base as a simple Mapping of Path to Content"""

#     class About(TypedDict):
#       """Metadata about the Project"""
#       Project: Ctx.Spec.About.ProjectStats
#       Python: Ctx.Spec.About.PythonStats
#       Git: Ctx.Spec.About.GitStats

#       class ProjectStats(TypedDict):
#         README: str

#         @staticmethod
#         def factory(worktree: pathlib.Path) -> Ctx.Spec.About.ProjectStats:
#           return {
#             'README': (worktree / 'README.md').read_text(),
#           }

#       class PythonStats(TypedDict):
#         Version: str

#         @staticmethod
#         def factory(worktree: pathlib.Path) -> Ctx.Spec.About.PythonStats:
#           return {
#             'Version': subprocess.run(['python', '--version'], check=True, capture_output=True).stdout.decode().strip(),
#           }

#       class GitStats(TypedDict):
#         Version: str
#         Branches: str
#         Log: str
#         HEAD: str

#         @staticmethod
#         def factory(worktree: pathlib.Path) -> Ctx.Spec.About.GitStats:
#           return {
#             'Version': subprocess.run(['git', '--version'], check=True, capture_output=True).stdout.decode().strip(),
#             'Branches': subprocess.run(['git', 'branch', '--list'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
#             'Log': 'Last 3 Commits:\n' + subprocess.run(['git', 'log', '--oneline', '-n', '3'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
#             'HEAD': subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, cwd=worktree.as_posix(), capture_output=True).stdout.decode().strip(),
#           }

if __name__ == '__main__':
  def test_marshal_unmarshal():
    # Create a sample Chat.Log
    original_log = Chat.Log(log=[
      Chat.Message.factory("User", "Hello, how are you?"),
      Chat.Message.factory("Assistant", "I'm doing great, thanks!\nHow about you?"),
      Chat.Message.factory("User", "I'm fine too. Let's discuss the project."),
    ])

    marshaled_data = Chat.Log.marshal(original_log)
    print(marshaled_data)
    unmarshaled_log = Chat.Log.unmarshal(marshaled_data)

    # Compare each message
    for original_msg, unmarshaled_msg in zip(original_log['log'], unmarshaled_log['log']):
      assert original_msg['publisher'] == unmarshaled_msg['publisher']
      assert original_msg['content'].strip() == unmarshaled_msg['content'].strip()
      assert original_msg['created'] == unmarshaled_msg['created']

    # Optional: Print the marshaled data for visual inspection
    print(
f"""
Marshaled Data:

####################

{marshaled_data.decode()}

####################
"""
    )  
  test_marshal_unmarshal()