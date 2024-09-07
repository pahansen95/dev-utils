"""

An Agent to Assist with development

K.I.S.S: Opt for the dumbest implementation first.

"""
from __future__ import annotations
import logging, pathlib, os, sys, tempfile, subprocess, shlex, re, hashlib
from typing import Literal, Callable
from collections import deque

logger = logging.getLogger(__name__)
is_stdin_tty = os.isatty(sys.stdin.fileno())

class SubCommands:

  @staticmethod
  def add_document(
    corpus_root: pathlib.Path,
    doc_name: str | None,
    doc_prefix: pathlib.Path | None,
    doc_src: str,
    doc_kind: str,
  ) -> int:
    """Add a Document into the Corpus
    
    Optionally specify a name, otherwise one will be generated.
    Optionally specify a path prefix, relative to the corpus root, to write the Document Root under.
    
    """
    import DevAgent.KnowledgeBase as kb

    try: corpus = kb.corpus.Corpus.factory(corpus_root)
    except:
      logger.error(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1
    
    if doc_name is not None and doc_name in corpus.documents:
      logger.error(f'Document Name already exists in the Corpus: {doc_name}')
      return 1

    ### Fetch the Document
    if doc_kind == 'text': doc_factory: Callable[..., kb.corpus.Document] = ...
    else:
      logger.error(f'Unknown document Kind: {doc_kind}')
    raw_doc = ... # TODO: Implement Document Fetching

    if not doc_name: raise NotImplementedError('Document Name Generation')

    corpus.documents[doc_name] = doc_factory(
      name=doc_name,
      src=raw_doc,
      path=doc_prefix or '',
    )

    corpus.save_document(doc_name)

  @staticmethod
  def analyze(
    corpus_root: pathlib.Path,
    document_name: str,
  ) -> int:
    """Analyze the file & update the cache semantics"""

    import DevAgent.KnowledgeBase as kb

    try: corpus = kb.corpus.Corpus.factory(corpus_root)
    except:
      logger.error(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    if document_name not in corpus.documents[document_name]:
      logger.error(f'Could not find the Document {document_name} in the corpus located at {corpus.root.as_posix()}')
      return 1

    document = corpus.documents[document_name]

    ### Build the set of expected Semantics.

    # First, Generate Embeddings for the Document
    doc_embeds = ...

    # Next, Summarize the Document
    doc_sum = ...

    # Then, Classify the Document
    doc_classify = ...

    ### Update the Document Semantics
    document.semantics = kb.corpus.DocumentSemantics(
      summary=doc_sum,
      classifications=doc_classify,
      embeddings=doc_embeds,
    )
    
    ### Write the updated Document to Disk
    corpus.save_document(document.metadata.name)

    ### TODO: Sync the Git Project?

  @staticmethod
  def search(
    corpus_root: pathlib.Path,
    query: str,
    threshold: float,
  ) -> int:
    """Search the Knowledge Base for Information relevant to a query

    """

    import DevAgent.KnowledgeBase as kb

    try: corpus = kb.corpus.Corpus.factory(corpus_root)
    except:
      logger.error(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1
    
    if not (search_results := kb.semantics.search(
      query=query,
      corpus=corpus,
      threshold=threshold,
    )):
      logger.error(f'No Results Found for query: {query}')
      return 1

    sys.stdout.write(kb.pretty_print(search_results))
    return 0

def _parse_flag(flag: str) -> tuple[str, str]:
  assert flag.startswith('-')
  if '=' in flag: return flag[1:].split('=', maxsplit=1)
  else: return flag[1:], True

def main(argv: list[str], env: dict[str, str]) -> int:
  logger.debug(f'ARGV: {argv}')
  args = deque(a for a in argv if not a.startswith('-'))
  logger.debug(f'Arguments: {args}')
  flags = dict(_parse_flag(f) for f in argv if f.startswith('-'))
  logger.debug(f'Flags: {flags}')
  subcmd = args.popleft()
  if subcmd.lower() == 'search': return SubCommands.search(
    ...
  )
  else: raise RuntimeError(f'Unknown Subcommand: {subcmd}')

if __name__ == '__main__':
  logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'), stream=sys.stderr)
  _rc = 2
  try:
    _rc = main(sys.argv[1:], os.environ)
  except SystemExit as e:
    _rc = e.code
  except RuntimeError as e:
    logger.critical(f'Error Encountered: {e}')
  except Exception as e:
    logger.exception(e)
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    exit(_rc)
