"""

Tree Diffs

"""
from __future__ import annotations
from .core import *
from .protocols import *

@dataclass
class Diff[T]:
  """A Diff between two sets of items"""

  common: frozenset[T]
  """Commonality between trees"""
  left: frozenset[T]
  """Uniqueness to the left tree"""
  right: frozenset[T]
  """Uniqueness to the right tree"""

@dataclass
class TreeDiff[EK]:
  """The Diff between two trees"""

  topology: TopologyDiff[EK]
  """Topological Difference"""
  semantics: SemanticDiff
  """Semantic Difference"""

@dataclass
class TopologyDiff[EK]:
  """Tree Diff with respect to it's topology"""

  nodes: Diff[ID_T]
  """Diff in Nodes present"""
  keys: Diff[ID_T]
  """Diff in the Keys present per Tree"""
  edges: dict[EK, Diff[Edge[EK]]]
  """Diff in Edges, grouped by Edge Key"""

@dataclass
class SemanticDiff:
  """Tree Diff with respect to it's semantics"""

  structural: Diff[ID_T]
  """The semantic difference scoped only to topologically identical nodes; ie. does common node `foo.bar` have the same semantic fingerprint"""
  aggregate: Diff[HASH_T]
  """The difference of the entire set of semantics contained per tree ignoring topology"""

def calc_diff[NT, EK](lt: Tree[NT, EK], rt: Tree[NT, EK]) -> TreeDiff[NT, EK]:
  """Calculates the primitive difference between two (ordered) trees.
  
  A tree has topology (ie. structure) & semantics (ie. meaning). Therefore, "difference" is slightly nuanced:

  Structural differences are simple node & edge set differences. This is because nodes & edges are the tree themselves,
  nodes identify locations while edges identify topological relationships.

  Semantic differences include both semantic fingerprint set differences & structural node semantic comparisons. This is because
  semantics cannot be (pragmatically) generalized; the knowledge system the tree encodes could potentially be sensitive to the
  structural location of semantic information for example. Higher Order semantic differences
  (ie. Transformations, Hierarchical Contextualization) are left to developer to implement per their semantic system.
  
  """
  topology_diff = calc_topology_diff(lt, rt)
  semantic_diff = calc_semantic_diff(lt, rt, topology_diff)
  return TreeDiff(
    topology=topology_diff,
    semantics=semantic_diff
  )

def calc_topology_diff[NT, EK](lt: Tree[NT, EK], rt: Tree[NT, EK]) -> TopologyDiff[EK]:
  """The difference in structure between two trees.

  The Topological Difference is two fold:

  - The Node Set Difference
  - The Edge Set Difference (grouped by key)

  Node Set Diff can be used to track addition or removal of nodes
  Edge Set Diff can be used to track sibling reordering or re-organization of a hierarchy.
  
  """
  _ln, _rn = lt.nodes, rt.nodes
  _le, _re = lt.edges, rt.edges

  ### Node Diff
  l_n = frozenset(_ln.keys())
  r_n = frozenset(_rn.keys())
  n_common = l_n.intersection(r_n)
  l_n_uniq = l_n - n_common
  r_n_uniq = r_n - n_common
  node_diff = Diff(n_common, l_n_uniq, r_n_uniq)

  ### Key Diff
  l_ek = frozenset(_le.keys())
  r_ek = frozenset(_re.keys())
  ek_common = l_ek.intersection(r_ek)
  l_ek_uniq = l_ek - ek_common
  r_ek_uniq = r_ek - ek_common
  key_diff = Diff(ek_common, l_ek_uniq, r_ek_uniq)
  
  ### Edge diff per key
  e_common = {}
  # Preload l & r uniq edges w/ their uniq keys
  l_e_uniq = { ek: frozenset(_le[ek]) for ek in l_ek_uniq }
  r_e_uniq = { ek: frozenset(_re[ek]) for ek in r_ek_uniq }
  for ek in ek_common:
    l_e = frozenset(_le[ek])
    r_e = frozenset(_re[ek])
    e_common[ek] = (_e_common := l_e.intersection(r_e))
    l_e_uniq[ek] = l_e - _e_common
    r_e_uniq[ek] = r_e - _e_common
  edge_diff = {
      ek: Diff(
        e_common.get(ek, frozenset()),
        l_e_uniq.get(ek, frozenset()),
        r_e_uniq.get(ek, frozenset())
      ) for ek in (l_ek | r_ek)
    }

  return TopologyDiff[EK](node_diff, key_diff, edge_diff)

def calc_semantic_diff[NT, EK](lt: Tree[NT, EK], rt: Tree[NT, EK], topo_diff: TopologyDiff[EK]) -> SemanticDiff:
  """The difference in semantic values between two trees.
  
  Semantic Difference is nuanced:

  - Aggregate Diff: The difference of all semantic values found in the tree ignoring their topology.
  - Structural Diff: The difference of semantic values between topologically similiar nodes.

  `Topologically Similiar` implies nodes having the same location IDs; but not necessarily
  having the same set of ancestors & predecessors.

  Keep in mind that semantics can repeat inside a tree (assigned to multiple nodes).
    
  """

  _ln, _rn = lt.nodes, rt.nodes

  ### Aggregate Diff
  # First we get all the semantic values in both trees
  l_v = frozenset(n.fingerprint for n in _ln.values() if n.value != STRUCTURAL)
  r_v = frozenset(n.fingerprint for n in _rn.values() if n.value != STRUCTURAL)
  # Calc Diff
  v_common = l_v.intersection(r_v)
  l_v_uniq = l_v - v_common
  r_v_uniq = r_v - v_common
  aggregate_diff = Diff(v_common, l_v_uniq, r_v_uniq)

  ### Structurally similar diff
  # Gather all structurally similiar nodes having a value
  l_sv = frozenset(
    (n.location, n.fingerprint)
    for n in _ln.values()
    if n.location in topo_diff.nodes.common and n.value != STRUCTURAL
  )
  r_sv = frozenset(
    (n.location, n.fingerprint)
    for n in rt.nodes
    if n in topo_diff.nodes.common and n.value != STRUCTURAL
  )
  # Then calculate the diff using set theory
  sv_common = l_sv.intersection(r_sv)
  l_sv_uniq = l_sv - sv_common
  r_sv_uniq = r_sv - sv_common
  structural_diff = Diff(
    frozenset(loc for loc, _ in sv_common),
    frozenset(loc for loc, _ in l_sv_uniq),
    frozenset(loc for loc, _ in r_sv_uniq),
  )

  return SemanticDiff(structural_diff, aggregate_diff)
