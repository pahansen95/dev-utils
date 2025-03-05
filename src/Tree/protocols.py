"""

Tree Comparisons

"""
from __future__ import annotations

from .core import *

ID_T = TypeVar('ID_T')
HASH_T = TypeVar('HASH_T')

class Node[T](Protocol):
  """Nodes are Uniquely Identifable (Hashable) by their location within the tree"""

  location: ID_T
  """The Node's Structural ID, locating it within the tree's hierarchical structure"""
  fingerprint: HASH_T | None
  """The Node's Semantic ID, identifying uniqueness of it's contained value; if the node is purely structal (ie. having no value) then it has no fingerprint"""
  value: T | STRUCTURAL
  """The Node's Value; can be STRUCTURAL to denote a node serving purely as some structural member of a tree."""
  props: MutableMapping
  """Metadata Properties"""

class Edge[EK](Protocol):
  """Edges are Uniquely Identifable (Hashable) by their key, participant nodes & direction"""
  points: tuple[ID_T, ID_T]
  """The two Vertices comprising the vector (u, v)"""
  key: EK
  """The Key of the Edge"""
  direction: bool | None
  """The Direction of the edge; true: u->v; false: u<-v; None: u<->v (ie. undirected)"""
  props: MutableMapping
  """Metadata Properties"""
  
class Tree[NT, EK](Protocol):
  """An Ordered Tree parameterized by Node Type & Edge Key"""

  nodes: MutableMapping[ID_T, Node[NT]]
  """Nodes yeyed on their location IDs"""
  edges: MutableMapping[EK, set[Edge[EK]]]
  """Edges grouped by topology key"""
  height: MutableMapping[EK, int]
  """The depth of the tree grouped by topology"""
  props: MutableMapping
  """Metadata Properties"""
  fingerprint: HASH_T
  """The topologically contextualized semantic fingerprint of this (sub)tree"""

  def add_edge_key(self, edge_key: EK):
    """Add a new key to the topology"""

  def add_node(self,
    loc: ID_T,
    val: NT = STRUCTURAL,
    fngpnt: HASH_T = None,
    **props
  ):
    """Add a node to the tree specifying the, otherwise one is generated
    
    Args:

      loc (ID_T): The Location Id of the Node
      val (NT): The Optional Node Value; otherwise it's a structural node
      fngpnt (HASH_T): The Value's Semantic Fingerprint; only if val is provided.
      **props: Node Properties to pass through
    
    """
  def add_edge(self,
    key: EK,
    vertices: tuple[ID_T, ID_T],
    direction: bool | None,
    **props
  ):
    """Add an edge to the tree, updating references accordingly"""  

  def insert_node(self,
    loc: str,
    parent: str | None = None,
    key: str | None = None,
    val: NT = STRUCTURAL,
    fngpnt: str = None,
    **props
  ):
    """Inserts a node into the tree & add an edge between it & it's parent."""

  def ancestors_of(self, loc: ID_T, key: EK = None) -> Iterator[ID_T]:
    """An Iterator over the topological ancestors of the node at location"""

  def depth_of(self, loc: ID_T, key: EK = None) -> int:
    """The Depth of a node in the toplogy filtered by the edge key"""

  def parent_of(self, loc: ID_T, key: EK = None) -> ID_T:
    """The topological parent of the node at the specified location"""

  def children_of(self, loc: ID_T, key: EK = None) -> Sequence[ID_T]:
    """The Topological children of the node at the specified location"""

  def walk(self,
    root: ID_T = None,
    key: str = None,
    depth: tuple[int, int] = None,
    mode: WALK_T = 'dfs:pre'
  ) -> Generator[ID_T, None | bool | ID_T, None]:
    """Walk the tree starting from the specificied root using the mode specified yielding nodes.

    Args:

      root (ID_T): The Structural ID of the Node to start from; if not specified starts from the implicit root node.
      depth (int): The range of depth to constrain the traversal; only yields nodes within the range; by default the entire height: (0, self.height)
    
    To control traversal, a value can be sent back:

      bool: False means prune; True means continue
      None: Same as True
      Node[T]: Jump to the Node provided & resume walking from there; first returned node is the node jumped to
    
    """

