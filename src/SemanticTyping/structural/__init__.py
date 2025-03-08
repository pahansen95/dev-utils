"""

Implements Features for Structural Typing (static duck typing)

> NOTE: Python provides it's own structural typing functionality via Protocols

In Structural Typing, objects are defined to be "of a type" if they are compatible with that types declared features. Contextualized to
Python, objects are compatible w/ one another if one implements the other's type-hinted signature (properties & methods). This provides
the capability for superset & subset types; types having a signature of a union-set of types or having a partial signature of a single type.

In Semantic Typing, Developers define `SemanticTypes` & `ComputationalTypes`. `SemanticTypes` declare abstract object signature that are
structural types. `ComputationalTypes` implement concrete Python objects which adhere to any set of `SemanticTypes`; it is illegal for a
`ComputationalType` to declare it's participation in a `SemanticType` but then to not implement it. To simplify things further, our structural
typing design also requires:

- signatures match in both name, name ordering & typehints
- python objects inherit from the `SemanticType` type hierarchy (though technically speaking this is not an enforceable
requirement due to the dynamic nature of Python).

What does this mean in practice? Chiefly:

- Semantic Types must be fully annotated.
- Computational Types must fully implement Semantic Types.
- At runtime semantic type rules are evaluated during class creation of computation types.

"""
from __future__ import annotations
from typing import Literal, Protocol
import ast, logging

logger = logging.getLogger()

def compare_signatures(l: ast.AST, r: ast.AST):
  """Runs a comparison between the signatures of the namespaces described by the left & right abstract syntax trees.
  
  Returns the (
    union_set,
    left_diff,
    right_diff,
  ) of the signatures.

  Signature Comparison includes:

  - Check for matching names & name ordering
  - Check for matching typehints for matching names

  """
  union, l_diff, r_diff = set(), set(), set()
  ( l_sig, r_sig ) = map(
    lambda c: c(), # Call
    map(signatures.NamespaceSignature, (l, r)), # Factory
  )
  logger.debug(l_sig)
  logger.debug(r_sig)
  assert l_sig.keys() == r_sig.keys()

  ### Calculate the union
  for k in l_sig.keys():
    l_set, r_set = frozenset(l_sig[k]), frozenset(r_sig[k])
    union.update(_union := l_set.union(r_set))
    l_diff.update(l_set.difference(_union))
    r_diff.update(r_set.difference(_union))
  
  return (union, l_diff, r_diff)

from . import signatures

  
