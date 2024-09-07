"""

A Collection of Documents

"""
from __future__ import annotations
import pathlib
from typing import Literal
from dataclasses import dataclass

@dataclass
class Corpus:
  """The Collection of Documents"""
  root: pathlib.Path
  """The Root of the Corpuse Tree"""
  documents: dict[str, Document]
  """The set of documents in the Corpus, keyed on name"""

  @classmethod
  def factory(cls, src: pathlib.Path) -> Corpus:
    if isinstance(src, pathlib.Path): ...
    else: raise TypeError(type(src))
  
  def save_document(self, name: str):
    """Save the Document to Disk"""

@dataclass
class Document:
  """A Document in the Corpus"""
  root: pathlib.Path
  """The Root of the Document Tree relative to the Corpus Root"""
  metadata: DocumentMetadata
  """Metadata about the document"""
  semantics: DocumentSemantics
  """The Pre-Computed Semantics about the document"""

@dataclass
class DocumentMetadata:
  """Metadata regarding the document"""
  name: str
  """The Document's unique name identifying it in the Corpus"""
  document_kind: Literal['text']
  """The Kind of document; current kinds include:

  - `text`: a single, unstructured blob of text data
  """
  created_at: int
  """The date as a UTC posix second timestamp when the Document was added to the corpus"""
  publisher: dict[str, str]
  """Labels identifying the publisher of the Document"""

@dataclass
class DocumentSemantics:
  """The Computed Semantics for a document"""
  summary: str
  """A Salient Summary of the Document"""
  classifications: ...
  """Classifiers grouping this document relative to the Corpus"""
  embeddings: ...
  """Vector Embeddings for the Document"""