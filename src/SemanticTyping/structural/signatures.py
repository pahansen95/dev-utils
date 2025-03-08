from __future__ import annotations
from typing import Unpack, TypeVar, Any
from collections.abc import Iterator, Generator, Sequence, Mapping
from dataclasses import dataclass, field, KW_ONLY
from collections import deque
import ast, logging

logger = logging.getLogger(__name__)

@dataclass
class StackFrame[T, P]:
  node: T
  props: P
  def __iter__(self) -> tuple[T, P]: return iter((self.node, self.props))

@dataclass
class Stack[T, P]:
  frames: deque[StackFrame[T, P]] = field(default_factory=deque)

  def __bool__(self) -> bool: return len(self.frames) > 0
  def __len__(self) -> int: return len(self.frames)
  def __iter__(self) -> Iterator[tuple[T, P]]: return ( (f.node, f.props) for f in self.frames )

  def insert(self, *frame: Unpack[tuple[T, P]]):
    """Push a frame onto the bottom of the stack."""
    self.frames.appendleft(StackFrame(*frame))

  def push(self, *frame: Unpack[tuple[T, P]]):
    """Push a frame onto the top of the stack"""
    self.frames.append(StackFrame(*frame))
  
  def pop(self) -> tuple[T, P]:
    """Pop a Frame off the stack"""
    return tuple(self.frames.pop())

AST = TypeVar('AST', bound=ast.AST, contravariant=True)
@dataclass
class ASTNode[AST]:
  node: AST
  ancestors: Sequence[AST] = field(default_factory=tuple)

  def __getattr__(self, name): return getattr(self.node, name)

  def children(self) -> Iterator[AST]: return (
    ASTNode(child, tuple([ *self.ancestors, self.node ]))
    for child in ast.iter_child_nodes(self.node)
  )

@dataclass
class ASTProps:
  root_depth: int
  local_depth: int = field(default=0)

@dataclass
class ASTVisitor:
  root: AST
  max_depth: int = None
  local_depth: int = 0

  def __iter__(self): return self()
  def __call__(self) -> Generator[tuple[ASTNode, ASTProps], bool | None, None]:
    logger.debug(f'Walking AST from {type(self.root).__name__}:{self.local_depth}')
    stack: Stack[ASTNode, ASTProps] = Stack(frames=[ StackFrame(*v) for v in [
      ( ASTNode(self.root), ASTProps(0, local_depth=self.local_depth) ),
    ] ])
    while stack:
      visit_node, node_props = stack.pop()
      assert isinstance(visit_node, ASTNode)
      add = yield visit_node, node_props
      if add is not None and not add: continue
      for child_node in reversed(list(self.visit(visit_node, node_props))):
        stack.push(child_node, ASTProps(
          node_props.root_depth + 1,
          node_props.local_depth + 1,
        ))

  def visit(self, node: ASTNode, props: ASTProps) -> Iterator[ASTNode]:
    # logger.debug(f'{type(node).__name__}:{depth}')
    _node = node.node
    logger.debug(f'{type(_node).__name__}:{props}\n{'\n'.join(f'\t{l}' for l in ast.unparse(_node).splitlines())}')
    _children = iter([])
    try: visit = getattr(self, f'visit_{type(_node).__name__}')
    except AttributeError: logger.warning(f'Unhandled AST Node: {type(_node).__name__}')
    else:
      if visit(node, props): _children = node.children()
    return _children
  
  def visit_ClassDef(self, node: ASTNode[ast.ClassDef], props: ASTProps) -> bool:
    if props.local_depth > 0: return False # TODO: don't visit nested classes unless it's an immediate child of a Module
    return True

  def visit_AnnAssign(self, node: ASTNode[ast.AnnAssign], props: ASTProps) -> bool:
    return False
  
  def visit_FunctionDef(self, node: ASTNode[ast.FunctionDef], props: ASTProps) -> bool:
    return False
      
  def visit_arguments(self, node: ASTNode[ast.arguments], props: ASTProps) -> bool:
    if props.local_depth > 0: return False # only traverse when the depth is 0
    args_info = []
    # Process positional arguments
    for arg in node.args:
      arg_name = arg.arg
      if arg_name.lower() in ('self', 'cls'): arg_type = "Self" # Force the "Self" Typehint
      else: arg_type = "Any" # Default if no annotation
      if arg.annotation: arg_type = ast.unparse(arg.annotation)
      args_info.append(f"{arg_name}: {arg_type}")
    # Process varargs (*args)
    if node.vararg:
      vararg_name = node.vararg.arg
      vararg_type = "Any"
      if node.vararg.annotation:
        vararg_type = ast.unparse(node.vararg.annotation)
      args_info.append(f"*{vararg_name}: {vararg_type}")
    # Process keyword-only arguments
    for kwonly in node.kwonlyargs:
      kwonly_name = kwonly.arg
      kwonly_type = "Any"
      if kwonly.annotation:
        kwonly_type = ast.unparse(kwonly.annotation)
      args_info.append(f"{kwonly_name}: {kwonly_type}")
    # Process kwargs (**kwargs)
    if node.kwarg:
      kwarg_name = node.kwarg.arg
      kwarg_type = "Any"
      if node.kwarg.annotation:
        kwarg_type = ast.unparse(node.kwarg.annotation)
      args_info.append(f"**{kwarg_name}: {kwarg_type}")
    
    logger.debug(f"  Node< {', '.join(args_info)} >")

SIG_T = dict[str, Sequence[ast.AST] | Mapping[str, ast.AST] | ast.AST]

class SignatureKind:
  node: ASTNode
  sig: dict
  def __str__(self) -> str: ...
  @staticmethod
  def is_a(node: ASTNode) -> bool: ...
  def extract(self) -> SIG_T: ...

@dataclass
class PropertySignature(SignatureKind):
  """
  A Property Signature is an annotated assignment.
  """
  node: ASTNode[ast.AnnAssign]
  sig: dict = field(init=False, default_factory=dict)
  def __str__(self) -> str:
    if self.sig['val'] is None: return f'Node< {ast.unparse(self.sig['name'])}: {ast.unparse(self.sig['ann'])} >'
    else: return f'Node< {ast.unparse(self.sig['name'])}: {ast.unparse(self.sig['ann'])} = {ast.unparse(self.sig['val'])} >'
  @staticmethod
  def is_a(node: ASTNode) -> bool: return isinstance(node.node, ast.AnnAssign)
  def extract(self) -> SIG_T:
    _node = self.node.node
    assert isinstance(_node, ast.AnnAssign)
    
    self.sig['name'] = _node.target
    self.sig['ann'] = _node.annotation
    self.sig['val'] = _node.value
    
    logger.debug(f'  {self}')
    return self.sig

@dataclass
class CapabilitySignature(SignatureKind):
  """
  A Capability Signature is an annotated FunctionDef or some Callable Object.
  """
  node: ASTNode
  sig: dict = field(init=False, default_factory=dict)
  @classmethod
  def is_a(self, node: ASTNode) -> bool:
    if isinstance(node.node, ast.FunctionDef): return True
    elif isinstance(node.node, ast.ClassDef):
      for node, _ in ASTVisitor(node.node, max_depth=1):
        _node = node.node
        if isinstance(_node, ast.ClassDef) and _node.name == '__call__': return True
        elif isinstance(_node, ast.Assign):
          for trgt in _node.targets:
            if isinstance(trgt, ast.Name) and trgt.id == '__call__': return True
    elif isinstance(node.node, ast.Assign) and len(node.node.targets) == 1:
      trgt = node.node.targets[0]
      assert isinstance(trgt, ast.Name)
      if isinstance(trgt.ctx, ast.Load):
        raise NotImplementedError
    return False
  def extract(self) -> SIG_T:
    _node = self.node.node
    if isinstance(_node, ast.FunctionDef):
      self.sig['name'] = _node.name
      self.sig['deco'] = _node.decorator_list
      self.sig['tparam'] = _node.type_params
      self.sig['args'] = _node.args
      self.sig['retn'] = _node.returns
    elif isinstance(_node, ast.ClassDef):
      raise NotImplementedError # TODO: Walk the AST & Extract the signature of the __call__ method
    elif isinstance(_node, ast.Assign):
      raise NotImplementedError # TODO: Walk the AST & Extract the signature
    else: raise TypeError(_node)
    return self.sig

SIG_KT = type[PropertySignature | CapabilitySignature]

class NamespaceSignature:
  """
  
  Extract the Namespace signature of an AST

  A Semantic Type declares:

  - Properties
  - Methods

  So relative to the namespace provided, we need to collect the following constructs.

  - Variable Definitions
  - Callable Definitions 

  Collecting these constructs follows these general order of operations:

  - Begin by traversing the Namespace's AST Depth First Pre-Order (Topological sort) up to a depth of 1 (inclusive).
  - For each Node, classify them as a Variable or a Callable.
    - Callables are either Annotated Function Definitions or some named object directly
      implements an Annotated `__call__` Function Definition.
    - Variables are Annotated Assignments (x: int[ = 1])
    - All other nodes are ignored & their branches pruned
  - Then for each classified node, their signatures are parsed:
    - For variables this includes their name & their type annotation
    - For callables this includes their name, annotated arguments, return annotation & parameterization.
      - NOTE: Decorators will be stashed but won't be used for Structural type comparisons.
      - NOTE: For objects implementing __call__, their name is of the object or definition implementing __call__.
  """

  def __new__(cls, root: AST):
    return cls(**cls.assemble_signature(
      cls.collect_nodes(root)
    ))
  
  @classmethod
  def collect_nodes(cls, root: AST) -> dict[SIG_KT, list[ASTNode]]:
    """Traverse the AST from the root, collecting any potential AST nodes participating in a namespaces signature"""
    sig_nodes: dict[str, list[ASTNode]] = {
      PropertySignature: [],
      CapabilitySignature: [],
    }
    for node, prop in ASTVisitor(root, max_depth=1):
      for signature in (
        PropertySignature,
        CapabilitySignature
      ):
        if signature.is_a(node):
          sig_nodes[signature].append(node)
          break
      else: logger.debug(f'Pruning AST Branch: {type(node.node).__name__}')
    return sig_nodes

  @classmethod
  def assemble_signature(cls, collection: dict[SIG_KT, list[ASTNode]]) -> NamespaceSignature:
    return {
      sig_kind: [ sig_kind(node).extract() for node in nodes ]
      for sig_kind, nodes in collection.items()
    }
  
if __name__ == '__main__':
  import inspect, json

  @dataclass
  class Box[T]:
    val: T
    def get(self, *args: int | float, **kwargs: Unpack[MapHints]) -> T: ...
    def set(self, val: T) -> None: ...
    @classmethod
    def wrap(cls, val: T) -> Box[T]: ...
    @staticmethod
    def size_of(t: type) -> int: ...
    foo = size_of

  logging.basicConfig(level='DEBUG')
  module = ast.parse('\n'.join(l.removeprefix('  ') for l in inspect.getsource(Box).splitlines()))
  signature = (extractor := NamespaceSignature(
    root=module.body[0],
  ))()
  print({
    key.__name__: [
      s_k
      for s_k in sigs
    ]
    for key, sigs in signature.items()
  })
  print(json.dumps({
    key.__name__: [
      {
        k: (
          v
            if isinstance(v, str) else
          str(v)
            if isinstance(v, ast.AST) else
          [ str(_v) for _v in v]
            if isinstance(v, Sequence) else
          { k: str(_v) for _v in v }
            if isinstance(v, Mapping) else
          f'NotImplementedError({type(v)})'
        )
        for k, v in sig.items()
      } for sig in sigs
    ]
    for key, sigs in signature.items()
  }, indent=2))
