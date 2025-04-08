from __future__ import annotations
from typing import *
from collections.abc import *
from types import *

from dataclasses import dataclass, field, KW_ONLY, fields
import logging, pickle, regex as re, networkx as nx, itertools, functools

logger = logging.getLogger(__package__ or __name__)

class BNFError(ValueError): ...
class LexError(BNFError): ...
class ParseError(BNFError): ...

class Match(NamedTuple):
  value: str
  start: int
  stop: int
  
_pascal_to_snake = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')

@functools.cache
def _terminal_name(cls: type[Terminal]) -> str:
  return _pascal_to_snake.sub('_', cls.__name__).upper()

@dataclass
class Terminal:
  """A Regular Expression Pattern to Match"""
  regex: str
  _: KW_ONLY
  priority: int = 0
  _pattern: re.Pattern | None = None

  @property
  def name(self) -> str: return _terminal_name(type(self))

  def __hash__(self):
    return hash((self.name, self.regex))

  def __repr__(self) -> str: return repr(self.regex)
  def __str__(self) -> str: return repr(self.regex)[1:-1]
  def compile(self):
    if self._pattern is not None: return
    self._pattern = re.compile(self.regex, flags=re.MULTILINE)
  def scan(self, txt: str, start = 0) -> Iterator[Match]:
    """Scan the text for matches of this terminal's pattern."""
    assert self._pattern is not None
    for match in self._pattern.finditer(txt, pos=start):
      yield Match(
        value=match.group(0),
        start=match.start(),
        stop=match.end()
      )
  def match(self, txt: str, start = 0) -> Match | None:
    assert self._pattern is not None
    match_ = self._pattern.match(txt, pos=start)
    if match_: return Match(
      value=match_.group(0),
      start=match_.start(),
      stop=match_.end()
    )
    else: return None

@dataclass(frozen=True)
class Token:
  """Represents a lexical token."""
  kind: type[Terminal]
  value: str
  start: int
  stop: int
  lineno: int
  colno: int

@dataclass
class TokenBuffer:
  src: Sequence[Token]
  """The buffer of tokens"""
  _: KW_ONLY
  tokens: list[Token] = field(default=None, init=False)
  """The current token stack; in reverse order from the source"""
  pos: int = 0
  """The current token position"""
  size: int = field(default=None, init=False)
  """The size of the source"""

  def __post_init__(self):
    self.tokens = list(reversed(self.src))
    self.size = len(self.tokens)

  @property
  def lineno(self) -> int: return self.tokens[-1].lineno
  @property
  def colno(self) -> int: return self.tokens[-1].colno
  
  def __len__(self) -> int: return len(self.tokens)

  def pop(self) -> Token:
    """Pop the next token updating the position"""
    tok = self.tokens.pop()
    self.pos += 1
    return tok
  def push(self, token: Token):
    """Push back the token updating the position"""
    self.tokens.append(token)
    self.pos -= 1
  def peek(self, count = 1) -> Token | tuple[Token]:
    """Peek at the next token without consuming it"""
    assert count > 0
    if count > len(self): raise IndexError(f"Cannot peek `{count}` tokens from a `{len(self)}` sized buffer")
    if count == 1: return self.tokens[-1]
    else: return tuple(reversed(self.tokens))[0:count]

def tree_view(parse_tree: ParseTree, root_id: ID_T = None, indent: str = "", is_last: bool = True) -> str:
  """
  Visualize a parse tree as a string, similar to the Unix 'tree' command.
  """
  graph = parse_tree.graph
  
  # If no root is specified, find the implicit root
  if root_id is None:
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
      return "Empty tree"
    grammars = list(graph.successors(roots[0]))
    assert len(grammars) == 1
    root_id = grammars[0]
    assert parse_tree.get_node(root_id)['name'] == 'GRAMMAR', parse_tree.get_node(root_id)
  
  # Get node data
  node_data = parse_tree.get_node(root_id)
  
  # Format the current node
  if node_data.get('kind') == Terminal:
    label = f"{node_data['name'].name}: {repr(node_data['props']['token'].value)[1:-1]}"
  else:
    label = node_data.get('name', 'Unknown')
  
  # Create the branch prefix
  prefix = ""
  if indent:
    prefix = "└── " if is_last else "├── "
  
  # Start with this node
  result = f"{indent}{prefix}{label}\n"
  
  # Get all children
  children = list(graph.successors(root_id))
  
  # Create new indent for children
  new_indent = indent
  if indent:
    new_indent += "  " if is_last else "│   "
  else:
    new_indent = "  "
  
  # Process all children
  for i, child in enumerate(children):
    child_is_last = (i == len(children) - 1)
    result += tree_view(parse_tree, child, new_indent, child_is_last)
  
  return result

ID_T = int
@dataclass
class ParseTree:
  _: KW_ONLY
  graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
  idx: ID_T = 0
  ekey: str = 'parse'

  def __post_init__(self):
    try: self.root
    except IndexError:
      self.create_node()
      self.root # Validate
  
  @property
  def root(self) -> ID_T:
    root_nodes = [ n for n, deg in self.graph.in_degree if deg == 0 ]
    if len(root_nodes) <= 0: raise IndexError('No Root Nodes Found')
    elif len(root_nodes) > 1: raise ValueError('Multiple Root Nodes Found')
    return root_nodes[0]

  def __str__(self): return tree_view(self)

  def create_node(self) -> ID_T:
    node_id = self.idx
    self.graph.add_node(node_id)
    self.idx += 1
    return node_id
  
  def get_node(self, node_id: ID_T):
    return self.graph.nodes(data=True)[node_id]
  
  def add_edge(self, parent: ID_T, *children: ID_T, ekey: str = None):
    if ekey is None: ekey = self.ekey
    assert parent in self.graph
    for child in children:
      assert child in self.graph
      self.graph.add_edge(parent, child, key=ekey)
  
  def add_branch(self, parent: ID_T, child: ParseTree):
    """Add the child tree as a branch to parent"""
    assert parent in self.graph
    assert self.ekey == child.ekey
    
    # Get the child's root
    child_root = child.root
    
    # Calculate offset to avoid ID conflicts
    offset = self.idx
    
    # Create a mapping for node IDs (where child_root maps to parent)
    mapping = {child_root: parent}
    for node in child.graph.nodes():
      if node == child_root: continue
      mapping[node] = node + offset
    
    # Add all nodes except the child's root
    for node, data in child.graph.nodes(data=True):
      if node == child_root: continue
      self.graph.add_node(mapping[node], **data)
    
    # Add all edges, using the mapping to handle connections
    for u, v, key, data in child.graph.edges(data=True, keys=True):
      self.graph.add_edge(mapping[u], mapping[v], key=key, **data)
    
    # Update the index
    self.idx += len(child.graph.nodes()) - 1  # -1 for the root node we didn't add
    
@dataclass
class Rule:
  """Represents a Grammatical Rule"""
  tree: ParseTree
  """The Parse tree to update"""
  buffer: TokenBuffer
  """The Buffer of Tokens"""
  _: KW_ONLY
  name: str = None

  @classmethod
  def from_rule(cls, other: Rule, **kwargs):
    return cls(**(
      { 'tree': other.tree, 'buffer': other.buffer } | kwargs
    ))
  
  def __post_init__(self):
    # Patch the name variable based on the class's actual name
    cls = type(self)
    if cls is Rule: return
    assert cls is not Rule and issubclass(cls, Rule)

    # Update the name attribute
    if self.name is None:
      cls.name = _pascal_to_snake.sub('_', cls.__name__).upper()
      self.name = cls.name
    
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]: raise NotImplementedError
  def __call__(self, parent: ID_T):
    """Evaluate the Rule; consuming tokens & adding it to the tree"""
    assert type(self) is not Rule
    logger.debug(f'\n===\n🟡🟡🟡 {self.name}<{id(self):x}> EVAL START 🟡🟡🟡\n<{self.peek().kind.name}>\n{self.peek().value}\n</{self.peek().kind.name}>\n===')
    try:
      # Now update the tree
      node_id = self.tree.create_node()
      node_data = self.tree.get_node(node_id)
      node_data['kind'] = Rule
      node_data['name'] = self.name
      node_data['props'] = {} # TODO: anything to add?
      self.tree.add_edge(parent, node_id)
      for child in self:
        if child is None: continue
        elif isinstance(child, Token):
          leaf_id = self.tree.create_node()
          leaf = self.tree.get_node(leaf_id)
          leaf['kind'] = Terminal
          leaf['name'] = child.kind
          leaf['props'] = {
            'token': child,
          }
          self.tree.add_edge(node_id, leaf_id)
        elif isinstance(child, Rule):
          # logger.debug(f'Descending into Child Rule: {child.name}')
          child(node_id)
        elif isinstance(child, ParseTree):
          self.tree.add_branch(node_id, child)
        else: raise TypeError(type(child))
    except:
      result_msg = f'❌❌❌ {self.name}<{id(self):x}> EVAL FAILED ❌❌❌'
      raise
    else:
      result_msg = f'✅✅✅ {self.name}<{id(self):x}> EVAL PASSED ✅✅✅'
    finally: logger.debug(f'\n===\n{result_msg}\n===')

  def consume(self) -> Token:
    token = self.buffer.pop()
    self.log(f'Consuming Token...\n<{token.kind.name}>\n{token.value}\n</{token.kind.name}>')
    if not self.is_empty(): self.log(f'Current Token...\n<{self.peek().kind.name}>\n{self.peek().value}\n</{self.peek().kind.name}>')
    else: self.log('EOF Reached')
    return token

  # Basic checks - return booleans, don't raise
  def is_empty(self) -> bool:
    """Check if the token buffer is empty"""
    # self.log(f'Checking if buffer is empty: {len(self.buffer) == 0}')
    return len(self.buffer) == 0

  def token_matches(self, token: Token, *kinds: str) -> bool:
    """Check if the token is of one of the expected kinds"""
    self.log(f'token_matches: Checking if token {token.kind.name} matches any of: {", ".join(kinds)}')
    result = token.kind.name in kinds
    self.log(
      f'Token {token.kind.name} found in ( {", ".join(kinds)} )'
        if result else
      f'Token {token.kind.name} not in ( {", ".join(kinds)} )'
    )
    # self.log(f'Token match result: {result}')
    return result

  def next_matches(self, *kinds: str, lookahead = 0) -> bool:
    """Check if the next token matches any of the specified kinds"""
    if self.is_empty():
      self.log(f'Buffer empty, next cannot match any of: {", ".join(kinds)}')
      return False
    
    token = self.peek(lookahead)
    result = self.token_matches(token, *kinds)
    return result
  
  # Error raising validators
  def raise_if_empty(self, message: str = None) -> None:
    """Raise ParseError if buffer is empty"""
    # self.log(f'Ensuring buffer is not empty')
    if self.is_empty(): 
      error_msg = message or f"Unexpected end of input in {self.name}"
      # self.log(f'Buffer empty, raising error: {error_msg}', level=logging.ERROR)
      self.parse_error(error_msg)
    # self.log(f'Buffer not empty, continuing')

  def raise_if_token_mismatch(self, token: Token, *kinds: str, message: str = None) -> None:
    """Raise ParseError if token doesn't match any expected kinds"""
    # self.log(f'Ensuring token {token.kind.name} matches any of: {", ".join(kinds)}')
    if not self.token_matches(token, *kinds):
      expected = ", ".join(f"'{k}'" for k in kinds)
      error_msg = message or f"Expected {expected}, got '{token.kind.name}' in {self.name}"
      self.log(f'Token mismatch, raising error: {error_msg}', level=logging.ERROR)
      self.parse_error(error_msg)
    # self.log(f'Token matches expected kinds, continuing')

  # Combined operations that use the simpler methods
  def peek(self, lookahead = 0) -> Token:
    """Get the next token without consuming it, raising if empty"""
    # self.log(f'Peeking at next token')
    self.raise_if_empty()
    if lookahead == 0: token = self.buffer.peek()
    else: token = self.buffer.peek(1 + lookahead)[-1]
    # self.log(f'Peeked token: {token.kind.name} with value "{token.value}"')
    return token

  def peek_expected(self, *kinds: str, lookahead = 0, message: str = None) -> Token:
    """Peek at next token, verifying it matches, raising if empty or wrong type"""
    self.log(f'Peeking at next token, expecting one of: {", ".join(kinds)}')
    token = self.peek(lookahead)
    self.raise_if_token_mismatch(token, *kinds, message=message)
    # self.log(f'Successfully peeked expected token: {token.kind.name}')
    return token

  def expect_token(self, *kinds: str, message: str = None) -> Token:
    """Expect a token of specified kind(s), consume and return it"""
    # self.log(f'Expecting and consuming token of kind(s): {", ".join(kinds)}')
    _ = self.peek_expected(*kinds, message=message)
    token = self.consume()
    # self.log(f'Consumed expected token: {token.kind.name} with value "{token.value}"')
    return token

  def optional_token(self, *kinds: str) -> Optional[Token]:
    """Consume token if present and matching, otherwise return None"""
    # self.log(f'Checking for optional token of kind(s): {", ".join(kinds)}')
    if self.is_empty(): 
      self.log(f'Buffer empty, optional token not found: {", ".join(kinds)}')
      return None
    
    token = self.peek()
    if self.token_matches(token, *kinds):
      consumed = self.consume()
      self.log(f'Optional token {consumed.kind.name} found in {", ".join(kinds)}')
      return consumed
    
    self.log(f'Optional token not found: {", ".join(kinds)}')
    return None

  def match_pattern(self, *kinds: str) -> bool:
    """Checks if the provided sequence matches the head of the buffer"""
    self.log(f'Matching pattern: {", ".join(kinds)}')
    try:
      tokens = self.buffer.peek(len(kinds))
      assert isinstance(tokens, tuple)
      token_kinds = tuple(t.kind.name for t in tokens)
      result = token_kinds == kinds
      self.log(f'Pattern match result: {result} (found {" ".join(token_kinds)})')
      return result
    except IndexError:
      self.log(f'Pattern match failed: not enough tokens in buffer')
      return False
  
  def expect_pattern(self, *kinds: str, message: str = None):
    self.log(f'Expecting pattern: {", ".join(kinds)}')
    if not self.match_pattern(*kinds):
      error_msg = message or f"Expected Pattern {kinds}"
      self.log(f'Pattern match failed, raising error: {error_msg}', level=logging.ERROR)
      self.parse_error(error_msg)
    self.log(f'Pattern match succeeded')
    return None

  def get_literal_value(self) -> str:
    """Get the value of a Literal Token"""
    self.log(f'Getting literal value from token')
    token = self.peek()
    self.raise_if_token_mismatch(token, 'LITERAL', 'ESCAPED_LITERAL')
    
    if token.kind.name == 'LITERAL': 
      value = token.value
    else:
      assert token.kind.name == 'ESCAPED_LITERAL'
      value = token.value.strip('`')
    
    self.log(f'Retrieved literal value: `{value}`')
    return value

  def is_literal_value(self, value: str) -> bool:
    """Check if a Literal Token's value matches"""
    self.log(f'Checking if literal value matches: "{value}"')
    actual_value = self.get_literal_value()
    result = actual_value == value
    self.log(f'Literal value match result: {result} (actual: "{actual_value}")')
    return result
  
  def expect_literal_value(self, value: str, message: str = None) -> Token:
    """Consume a Literal Token if the value matches"""
    self.log(f'Expecting literal value: "{value}"')
    if not self.is_literal_value(value):
      actual_value = self.get_literal_value()
      error_msg = message or f"Expected Literal Value `{value}` but got `{actual_value}`"
      self.log(f'Literal value mismatch, raising error: {error_msg}', level=logging.ERROR)
      self.parse_error(error_msg)
    
    token = self.consume()
    self.log(f'Consumed token with expected literal value: "{value}"')
    return token

  # State management methods
  def save_state(self) -> Tuple[int, List[Token]]:
    """Save current buffer state for backtracking"""
    # self.log(f'Saving parser state at position {self.buffer.pos}')
    return (self.buffer.pos, list(self.buffer.tokens))

  def load_state(self, state: Tuple[int, List[Token]]) -> None:
    """Restore buffer to previously saved state"""
    pos, tokens = state
    # self.log(f'Restoring parser state from position {self.buffer.pos} to {pos}')
    self.buffer.pos = pos
    self.buffer.tokens = tokens

  # Error handling
  def message(self, body: str) -> str:
    if len(self.buffer) == 0: header = 'EOF'
    else: header = f'Ln {self.buffer.lineno}, Col {self.buffer.colno}'
    return f'{header} - {self.name}<{id(self):x}>: {body}'

  def parse_error(self, message: str) -> None:
    """Raise ParseError with current position information"""
    raise ParseError(self.message(message))
  
  def log(self, message: str, level: int = logging.DEBUG) -> None:
    logger.log(level, self.message(message))

  # Parsing helpers with state management
  def try_parse(self, rule_func: Callable[[], Any]) -> bool:
    """
    Try to parse using a rule function.
    Returns True if successful, False otherwise.
    Restores buffer state on failure.
    """
    self.log(f'Attempting to parse with rule function')
    state = self.save_state()
    try:
      rule_func()
      self.log(f'Parse attempt succeeded')
      return True
    except ParseError as e:
      self.log(f'Parse attempt failed: {e}')
      self.load_state(state)
      return False

  def try_with_result[T](self, rule_func: Callable[[], T]) -> Tuple[bool, Optional[T]]:
    """
    Try to execute a parsing function, capturing its result.
    Returns (success, result) tuple.
    Restores buffer state on failure.
    """
    self.log(f'Attempting to parse with rule function and capture result')
    state = self.save_state()
    try:
      result = rule_func()
      self.log(f'Parse attempt with result succeeded')
      return True, result
    except ParseError as e:
      self.log(f'Parse attempt with result failed: {e}')
      self.load_state(state)
      return False, None 

  def try_rule(self,
    rule_type: type[Rule],
    initial_state: tuple[int, list[Token]],
  ) -> tuple[type[Rule], ParseTree, tuple[int, list[Token]]] | None:
    self.log(f'Attempting to parse with rule type: {rule_type.__name__}')
    tree = ParseTree()
    rule = rule_type(tree, self.buffer)
    self.log(f'Evaluating `{rule.name}` SubRule')
    try:
      rule(tree.root)
    except ParseError as e:
      self.log(f'Failed evaluating `{rule.name}` SubRule: {e}')
      return None
    else:
      self.log(f'Successfully evaluated `{rule.name}` SubRule with {tree.graph.order()} nodes')
      return rule_type, tree, self.save_state()
    finally:
      self.load_state(initial_state)
  
  def collect_rule(
    rule_type: type[Rule],
    initial_state: tuple[int, list[Token]],
  ) -> list | None:
    raise NotImplementedError
  
