"""

Provides Contextual Datastructures & Utilities for the Agent

"""
from __future__ import annotations
from typing import TypedDict, NotRequired, Callable
import pathlib, logging, importlib.util

logger = logging.getLogger(__name__)

FileMap_t = dict[pathlib.Path, str]

class Spec(TypedDict):
  about: Spec.About
  """Metadata about the Project"""
  src: NotRequired[FileMap_t]
  """The Project Source as a FileMap"""
  docs: NotRequired[FileMap_t]
  """The Project Documentation as a FileMap"""
  other: NotRequired[list[Spec.Entry]]
  """User Defined Context Entries"""

  class Entry(TypedDict):
    title: str
    content: NotRequired[str]
    files: NotRequired[FileMap_t]

  class About(TypedDict):
    Project: Spec.About.ProjectStats
    Python: Spec.About.PythonStats
    Git: Spec.About.GitStats

    class ProjectStats(TypedDict):
      README: str

    class PythonStats(TypedDict):
      Version: str

    class GitStats(TypedDict):
      Version: str
      Branches: str
      Log: str
      HEAD: str

def load_spec_factory(filepath: pathlib.Path) -> Callable[..., Spec]:
  """Load the `spec_factorty` function from the given file"""
  module_spec = importlib.util.spec_from_file_location('spec_module', filepath)
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  return module.spec_factory

### Package Includes to avoid Cyclic Imports
from . import CtxUtils
###