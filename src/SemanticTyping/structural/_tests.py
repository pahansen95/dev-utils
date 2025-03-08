from __future__ import annotations
from typing import Sequence, Mapping
import ast, inspect, json, logging, io
from dataclasses import dataclass

if __name__ == '__main__':
  logger = logging.getLogger(__name__)
  logging.basicConfig(level='DEBUG')

class FakeModule:

  @classmethod
  def as_module(cls) -> ast.Module:
    fake_module = ast.parse(inspect.getsource(cls)).body[0]
    src = io.StringIO()
    for child in ast.iter_child_nodes(fake_module):
      if isinstance(child, ast.ClassDef):
        src.write(ast.unparse(child))
        src.write('\n')
    logger.debug(f'Source code for fake module {cls.__name__}...\n' + src.getvalue())
    return ast.parse(src.getvalue())

### Test Structural Signatures
class StructuralSignatureExctraction(FakeModule):
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
  
  @classmethod
  def test(cls):
    from SemanticTyping.structural.signatures import NamespaceSignature
    module = cls.as_module()
    signature = (extractor := NamespaceSignature(
      root=module.body[0],
    ))()
    # print({
    #   key.__name__: [
    #     s_k
    #     for s_k in sigs
    #   ]
    #   for key, sigs in signature.items()
    # })
    logger.info(json.dumps({
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

### Test Structural Comparisons
class StructuralComparison(FakeModule):
  class Box[T]:
    val: T
    def get(self) -> T: ...
    def set(self, val: T): ...
  
  class Result[T, E]:
    val: T | None
    err: E | None
    def get(self) -> tuple[T, None] | tuple[None, E]: ...
    def set(self, val: T): ...
    def set_err(self, err: E): ...

  @classmethod
  def test(cls):
    from SemanticTyping.structural import compare_signatures
    module = cls.as_module()
    assert len(module.body) == 2, len(module.body)
    (union, l_diff, r_diff) = compare_signatures(*module.body)

if __name__ == '__main__':
  for cls in (
    StructuralSignatureExctraction,
    StructuralComparison, 
  ):
    logger.info(f'Testing {cls.__name__}')
    cls.test()
