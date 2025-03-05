"""

Reference Implementation of a Tree

"""
from __future__ import annotations
from .core import *
from . import protocols as proto

@dataclass
class TreeNode[T](proto.Node[T]):
  location: str
  """The Node's Structural ID, locating it within the tree's hierarchical structure"""
  fingerprint: str | None = field(default=None)
  """The Node's Semantic ID, identifying uniqueness of it's contained value; if the node is purely structal (ie. having no value) then it has no fingerprint"""
  value: T | STRUCTURAL = field(default=STRUCTURAL)
  """The Node's Value; can be STRUCTURAL to denote a node serving purely as some structural member of a tree."""
  props: dict = field(default_factory=dict)
  """Metadata Properties"""

@dataclass
class TreeEdge(proto.Edge[str]):
  points: tuple[str, str]
  """The two node vertices of the vector (u, v) identified by their locations"""
  key: str
  """The Key of the Edge"""
  props: dict = field(default_factory=dict)
  """Metadata Properties"""
  _: KW_ONLY
  direction: Literal[True] = field(init=False, default=True)
  """Tree Edges are Directed u -> v"""

@dataclass
class OrderedMultiTree[NT](proto.Tree[NT, str]):
  
  height: dict[str, int] = field(default_factory=dict)
  """The Height of the Tree per edge set: { edge_key: height }"""
  props: dict = field(default_factory=dict)
  """Metadata Properties"""
  _: KW_ONLY

  _nodes: dict[str, TreeNode[NT]] = field(init=False, default_factory=dict)
  """Nodes available in the tree: { node_location: node }"""
  _edges: dict[str, set[TreeEdge]] = field(init=False, default_factory=dict)
  """Edges available in the tree, grouped by edge keys: { edge_key: set[edge] }"""
  _parents: dict[str, dict[str, str]] = field(init=False, default_factory=dict)
  """Node Parents: { edge_key: { node_location: parent_node_location } }"""
  _adjacent: dict[str, dict[str, list[str]]] = field(init=False, default_factory=dict)
  """Node Children; grouped by edge keys & ordered by insertion: { edge_key: { node_location: sorted([ node_location ]) } }"""
  _root_loc: str = field(default='')
  """The Node Location of the Implicit Root Node"""
  _default_ekey: str = field(default='primary')
  """The Default Edge Key"""

  @property
  def nodes(self) -> dict[str, TreeNode[NT]]: return { # filter out Implicit Root
    k: v for k, v in self._nodes.items()
    if k != self._root_loc
  }
  @property
  def edges(self) -> dict[str, set[TreeEdge]]: return {
    k: set( # Filter out all edges to the implicit root
      e for e in es
      if e.points[0] != self._root_loc
    )
    for k, es in self._edges.items()
  }
  @property
  def fingerprint(self) -> str:
    """Calculates a unique fingerprint based on the current set of location & semantic fingerprint pairs; can be used to directly compare two trees for equality"""
    return hex(hash(self))

  def __post_init__(self):
    assert self._root_loc not in self._nodes
    # Create immutable root node
    self._nodes[self._root_loc] = TreeNode(
      location=self._root_loc,
      value=STRUCTURAL,
      props={'implicit_root': True, 'mutable': False}
    )
    assert self._default_ekey not in self._edges.keys()
    self._edges[self._default_ekey] = set()
    self._parents[self._default_ekey] = { self._root_loc: self._root_loc }
    self._adjacent[self._default_ekey] = { self._root_loc: [] }
  
  def add_edge_key(self, edge_key: str):
    if edge_key in self._edges.keys(): raise ValueError(f'Edge Key already exists: {edge_key}')
    self.height[edge_key] = 0
    self._edges[edge_key] = set()
    self._parents[edge_key] = { self._root_loc: self._root_loc }
    self._adjacent[edge_key] = { self._root_loc: [] }

  def add_node(self,
    loc: str,
    val: NT = STRUCTURAL,
    fngpnt: str = None,
    **props
  ):
    if loc in self._nodes: raise ValueError(f'Node {loc} already exists')
    self._nodes[loc] = TreeNode(
      loc,
      fingerprint=fngpnt,
      value=val,
      props=props
    )

  def add_edge(self,
    key: str,
    vertices: tuple[str, str],
    direction: bool | None,
    **props,
  ):
    ### Add the Edge
    if not isinstance(direction, bool) and direction: raise ValueError(f'Trees only support U -> V Edges')
    if key not in self._edges.keys(): raise ValueError(f'Edge Key does not exist: `{key}`')
    if not all(n in self._nodes for n in vertices): raise ValueError(f'Undefined Node in `{vertices}`')
    edge = TreeEdge(points=vertices, key=key, props=props)
    if edge in self._edges[key]: raise ValueError(f'{edge} already exists')
    self._edges[key].add(edge)

    ### Update the topology references
    p_loc, loc = vertices
    self._parents[key][loc] = p_loc
    self._adjacent[key][p_loc].append(loc)
    # Calculate the new Depth
    depth = self.depth_of(loc, key)
    if depth > self.height[key]: self.height[key] = depth
  
  def ancestors_of(self, loc: str, key: str = None) -> Iterator[str]:
    if loc not in self._nodes: raise ValueError(f'Uknown Node: {loc}')
    if key is None: key = self._default_ekey
    if key not in self._edges.keys(): raise ValueError(f'Uknown Edge Key: {key}')
    if loc not in self._parents[key]: raise ValueError(f'Node does not have an edge in the `{key}` topology')
    _parents = self._parents[key]
    assert loc in _parents
    n, p = loc, _parents[loc]
    depth = 0
    while p != self._root_loc:
      depth += 1
      yield p
      n = p
      assert n in _parents
      p = _parents[n]
      if depth > 1_000_000: raise RuntimeError('Cycle Detected; depth exceeds 1,000,000') # TODO: Proper Cycle Detection

  def depth_of(self, loc: str, key: str = None) -> int:
    return len(list(self.ancestors_of(loc, key))) - 1 # Don't include the Implicit Root in the count

  def parent_of(self, loc: str, key: str = None) -> str:
    if loc not in self._nodes: raise ValueError(f'Uknown Node: {loc}')
    if key is None: key = self._default_ekey
    if key not in self._edges.keys(): raise ValueError(f'Uknown Edge Key: {key}')
    if loc not in self._parents[key]: raise ValueError(f'Node does not have an edge in the `{key}` topology')
    return self._parents[key][loc]
  
  def children_of(self, loc: str, key: str = None) -> list[str]:
    """The Ordered children of a Node in the topology filtered by the edge key."""
    if loc not in self._nodes: raise ValueError(f'Uknown Node: {loc}')
    if key is None: key = self._default_ekey
    if key not in self._edges.keys(): raise ValueError(f'Uknown Edge Key: {key}')
    if loc not in self._adjacent[key]: raise ValueError(f'Node does not have an edge in the `{key}` topology')
    return self._adjacent[key][loc]

  def insert_node(self,
    loc: str,
    parent: str | None = None,
    key: str | None = None,
    val: NT = STRUCTURAL,
    fngpnt: str = None,
    **props
  ):
    """Insert a node into the tree at the specified location."""
    if parent is None: parent = self._root_loc
    if loc == parent: raise ValueError(f'Cycle Detected: Cannot parent a node to itself: `{parent}` -> `{loc}`')
    if key is None: key = self._default_ekey
    if key not in self._edges.keys(): raise ValueError(f'Undefined Edge Key: {key}')
    if val is not STRUCTURAL and fngpnt is None: raise ValueError('Must specify a semantic fingerprint for the node value')
    if loc in self._nodes: raise ValueError(f'Node already exists at {loc}')
    if parent not in self._nodes: raise ValueError(f'Parent Node `{parent}` does not exist.')
    ### Add the Node & Edge
    _props = { 'mutable': True } | props
    self.add_node(
      loc, val, fngpnt,
      **_props
    )
    self.add_edge(
      key,
      (parent, loc),
      True,
      **_props,
    ) # This also updates topology references

  _walk_frame = tuple[str, int]
  """A Frame in the stack used by walk: ( node_loc, node_depth )"""

  def walk(self,
    root: str = None,
    key: str = None,
    depth: tuple[int, int] = None,
    mode: WALK_T = 'dfs:pre'
  ) -> Generator[str, None | bool | str, None]:
    if root is None: root = self._root_loc
    if root not in self._nodes: raise ValueError(f'Undefined Node: {root}')
    if key is None: key = self._default_ekey
    if key not in self._edges.keys(): raise ValueError(f'Undefined Edge Key: {key}')
    height = self.height[key]
    if depth is None: depth = (0, height)
    if depth[0] > depth[1]: raise ValueError(f'Malformed Depth; lowerbounds greater than upperbounds: {depth}')
    if depth[0] < 0 or depth[1] > height: raise ValueError(f'Depth out of bounds; expected range 0 <= h <= {height}; got {depth[0]} <= h <= {depth[1]}')
    if root == self._root_loc: root_depth = 0 # Override the Implicit Root
    else: root_depth = self.depth_of(root, key)
    if root_depth < depth[0] or root_depth > depth[1]: raise ValueError(f'Root depth out of bounds; expected range 0 <= h <= {height}; got root_depth')
    
    # TODO: Breakout into individual functions
    if mode == 'dfs:pre':
      adjacent = self._adjacent[key]
      stack: list[OrderedMultiTree._walk_frame] = []
      # If we start from the Implicit root, then prepopulate the stack w/ it's children (the real roots)
      if root == self._root_loc: stack.extend(*(
        ( real_root, root_depth )
        for real_root in reversed(adjacent[root]) # Reverse insertion order to maintain pop order
      ))
      else: stack.append(( root, root_depth ))
      def not_empty(): return len(stack) != 0
      def push(*frames: OrderedMultiTree._walk_frame): stack.extend(frames)
      def pop(): return stack.pop()
      while not_empty():
        node, node_depth = pop()
        assert node_depth >= depth[0] and node_depth <= depth[1]
        flow = yield node
        # Determine if the caller instructed us to prune a branch
        if isinstance(flow, bool):
          if not flow: continue
        elif isinstance(flow, str):
          # Drop all references to the Stack
          del pop, push, not_empty, stack
          yield from self.walk(root=flow, key=key, depth=depth, mode=mode) # Recurse
          return # Stop Iteration
        elif not (flow is None): raise TypeError(type(flow).__name__)
        # Default Case is to add children to the stack
        child_depth = node_depth + 1
        if child_depth > depth[1]: continue # Skip adding children out of bounds
        else: push(*( # Push in reverse order to pop in proper order
          (child_node, child_depth)
          for child_node in reversed(adjacent[node])
        ))

    else: raise ValueError(f'Undefined Walk Mode: {mode}')
