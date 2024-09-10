"""

An Agent to Assist with development

K.I.S.S: Opt for the dumbest implementation first.

"""
from __future__ import annotations
import logging, pathlib, os, sys, tempfile, subprocess, shlex, re, hashlib, itertools, requests, json, math
from typing import Literal, Callable, TextIO, BinaryIO
from collections import deque
from collections.abc import Iterator

logger = logging.getLogger(__name__)
is_stdin_tty = os.isatty(sys.stdin.fileno())

chunk_size = int(64 * 1024)

class Fetch:
  class Text:
    @staticmethod
    def filelike(f: TextIO | BinaryIO, chunk_size: int) -> Iterator[str]:
      chunk = f.read(chunk_size)
      if isinstance(chunk, bytes):
        _read = lambda chunk_size: f.read(chunk_size).decode()
        chunk = chunk.decode()
      elif isinstance(chunk, str): _read = lambda chunk_size: f.read(chunk_size)
      else: raise TypeError(chunk)
      while chunk:
        yield chunk
        chunk = _read(chunk_size)

    @staticmethod
    def fd(n: int, chunk_size: int = chunk_size) -> Iterator[str]:
      with open(n) as f:
        yield from Fetch.Text.filelike(f, chunk_size)
    @staticmethod
    def stdin(chunk_size: int = chunk_size) -> Iterator[str]:
      yield from Fetch.Text.filelike(sys.stdin, chunk_size)
    @staticmethod
    def file(path: str, chunk_size: int = chunk_size) -> Iterator[str]:
      with open(path) as f:
        yield from Fetch.Text.filelike(f, chunk_size)

    @staticmethod
    def http(url: str, chunk_size: int = chunk_size) -> Iterator[str]:
      with requests.get(url, headers={ 'Accept': 'text/plain;charset=utf-8' }, stream=True) as resp:
        resp.raise_for_status()
        yield from resp.iter_content(chunk_size=chunk_size, decode_unicode=True)

class SubCommand:

  @staticmethod
  def add_document_to_corpus(
    corpus_root: pathlib.Path,
    doc_name: str,
    doc_prefix: pathlib.Path | None,
    doc_src: str,
    doc_kind: str,
  ) -> int:
    """Add a Document into the Corpus
    
    Optionally specify a path prefix, relative to the corpus root, to write the Document Root under.
    
    """
    import DevAgent.KnowledgeBase as kb

    logger.debug("Assembling Corpus")
    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Error Loading the Corpus at {corpus_root.as_posix()}')
      return 1
    
    if doc_name in corpus.documents:
      logger.error(f'Document Name already exists in the Corpus: {doc_name}')
      return 1

    ### Fetch the Document
    logger.debug('Get Document Fetcher')
    _scheme, _remain = doc_src.split(':', maxsplit=1)    
    match (doc_kind, _scheme, _remain):
      case ('text', 'file', _): fetch_doc = Fetch.Text.file(_remain)
      case ('text', 'fd', '0'): fetch_doc = Fetch.Text.stdin()
      case ('text', 'fd', _): fetch_doc = Fetch.Text.fd(int(_remain))
      case ('text', 'http', _) | ('text', 'https', _): fetch_doc = Fetch.Text.http(doc_src)
      case ('text', _):
        logger.error(f'Unsupported `{doc_kind}` Document Source Scheme: {_scheme}')
        return 1
      case _:
        logger.error(f'Unsupported Document Kind: {doc_kind}')
        return 1

    logger.debug('Creating Document in Corpus')
    try:
      corpus.create_document(
        name=doc_name,
        kind=doc_kind,
        data=fetch_doc,
        prefix=doc_prefix,
      )
    except:
      logger.exception('Failed to Add the Document')
      if doc_name in corpus.documents: corpus.delete_document(doc_name)
      return 1
    
    logger.info(f'Successfully Added Document {doc_name} to Corpus')
    return 0

  @staticmethod
  def validate_corpus(
    corpus_root: pathlib.Path,
  ) -> int:
    """Validate the Corpus; useful for validating documents after manual patching"""

    import DevAgent.KnowledgeBase as kb

    logger.debug('Building Corpus')
    try: kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    logger.info('Corpus is Valid')
    return 0

  @staticmethod
  def generate_doc_embeddings(
    corpus_root: pathlib.Path,
    document_name: str,
    chunk_size: int,
    batch_size: int,
  ) -> int:
    """Generate Embeddings for a Document in the Corpus"""

    import DevAgent.KnowledgeBase as kb

    logger.info('Loading Corpus')
    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    logger.info('Loading Document')
    if document_name not in corpus.documents:
      logger.error(f'Could not find the Document {document_name} in the corpus located at {corpus.root.as_posix()}')
      return 1
    document = corpus.documents[document_name]

    ### Build the set of expected Semantics.
    logger.debug(f'Chunk Size: {chunk_size}, Batch Size: {batch_size}')

    doc_embeds: list[kb.embed.Embedding] = []

    # Process the document in batches
    doc_idx = 0
    for idx, chunks in enumerate(itertools.batched(
      corpus.read_document(document_name, chunk=chunk_size),
      batch_size
    )):
      assert isinstance(chunks, tuple) and all(isinstance(c, str) for c in chunks)
      logger.info(f'Processing Batch {idx+1}')
      chunk_embeds = kb.Semantics.embed(*chunks)
      ### NOTE: We need to track what chunk each embedding correlates to
      ### To calculate the starting & ending Index of each chunk in the batch
      ### We nee
      chunk_indices: list[dict[str, int]] = []
      for _ in range(len(chunks)):
        chunk_indices.append({
          'start': doc_idx,
          'len': chunk_size,
        })
        doc_idx += chunk_size
      chunk_embeds['metadata'] = {
        'DocChunks': chunk_indices
      }
      ###
      assert isinstance(chunk_embeds, dict)
      doc_embeds.append(chunk_embeds)
      ### Sync the Partial Results
      document.semantics['embeddings'] = doc_embeds
      corpus.update_document(document.metadata['name'])
    
    ### TODO: Sync the Git Project?

  @staticmethod
  def distill_doc(
    corpus_root: pathlib.Path,
    document_name: str,
  ) -> int:
    """Distill a Document in the Corpus extracting salient information"""

    import DevAgent.KnowledgeBase as kb

    logger.info('Loading Corpus')
    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    logger.info('Loading Document')
    if document_name not in corpus.documents:
      logger.error(f'Could not find the Document {document_name} in the corpus located at {corpus.root.as_posix()}')
      return 1
    document = corpus.documents[document_name]

    ### Build the set of expected Semantics.

    # TODO: align Chunk Size & Batch Size relative to the document size
    chunk_size = int(1024 * 4)
    batch_size = 8
    logger.debug(f'Chunk Size: {chunk_size}, Batch Size: {batch_size}')

    doc_distillate: set[str] = set()

    # Process the document in batches
    for idx, chunks in enumerate(itertools.batched(
      corpus.read_document(document_name, chunk=chunk_size),
      batch_size
    )):
      logger.info(f'Processing Batch {idx+1}')
      chunk_distillate = kb.Semantics.distill(*chunks, existing=list(doc_distillate))
      assert all(isinstance(e, str) for e in chunk_distillate)
      doc_distillate.update(chunk_distillate)
      ### Sync the Partial Results
      document.semantics['distillate'] = list(doc_distillate)
      corpus.update_document(document.metadata['name'])
    
    ### Write the updated Document to Disk
    corpus.update_document(document.metadata['name'])

    ### TODO: Sync the Git Project?

  @staticmethod
  def summarize_doc(
    corpus_root: pathlib.Path,
    document_name: str,
  ) -> int:
    """Summarize a Document in the Corpus; requires Distillate to be present"""

    import DevAgent.KnowledgeBase as kb

    logger.info('Loading Corpus')
    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    logger.info('Loading Document')
    if document_name not in corpus.documents:
      logger.error(f'Could not find the Document {document_name} in the corpus located at {corpus.root.as_posix()}')
      return 1
    document = corpus.documents[document_name]

    ### Build the set of expected Semantics.

    # TODO: align Chunk Size & Batch Size relative to the document size
    chunk_size = int(1024 * 4)
    batch_size = 8
    logger.debug(f'Chunk Size: {chunk_size}, Batch Size: {batch_size}')

    doc_distillate: list[str] = document.semantics.get('distillate', [])
    if not doc_distillate:
      logger.error(f'Document {document_name} does not have a Distillate')
      return 1
    
    ### Generate the Document Summary
    document.semantics['summary'] = kb.Semantics.summarize(*doc_distillate)
    
    ### Write the updated Document to Disk
    corpus.update_document(document.metadata['name'])

    ### TODO: Sync the Git Project?
  
  @staticmethod
  def distill_doc(
    corpus_root: pathlib.Path,
    document_name: str,
    chunk_size: int,
    batch_size: int,
  ) -> int:
    """Distill a Document in the Corpus extracting salient information"""

    import DevAgent.KnowledgeBase as kb

    logger.info('Loading Corpus')
    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1

    logger.info('Loading Document')
    if document_name not in corpus.documents:
      logger.error(f'Could not find the Document {document_name} in the corpus located at {corpus.root.as_posix()}')
      return 1
    document = corpus.documents[document_name]

    ### Build the set of expected Semantics.

    logger.debug(f'Chunk Size: {chunk_size}, Batch Size: {batch_size}')

    doc_distillate: set[bytes] = set()

    # Process the document in batches
    for idx, chunks in enumerate(itertools.batched(
      corpus.read_document(document_name, chunk=chunk_size),
      batch_size
    )):
      logger.info(f'Processing Batch {idx+1}')
      chunk_distillate = kb.Semantics.distill(*chunks, existing=list(doc_distillate))
      assert all(isinstance(e, str) for e in chunk_distillate)
      doc_distillate.update(chunk_distillate)
      ### Sync the Partial Results
      document.semantics['distillate'] = list(doc_distillate)
      corpus.update_document(document.metadata['name'])
    
    ### Write the updated Document to Disk
    corpus.update_document(document.metadata['name'])

    ### TODO: Sync the Git Project?

  @staticmethod
  def search_corpus(
    corpus_root: pathlib.Path,
    query: str,
    similarity: tuple[int, int],
  ) -> int:
    """Search the Knowledge Base for Information relevant to a query

    """

    import DevAgent.KnowledgeBase as kb

    try: corpus = kb.Corpus.factory(corpus_root)
    except:
      logger.exception(f'Failed to find the Corpus at {corpus_root.as_posix()}')
      return 1
    
    logger.info(f'Search Similarity Bounds (Radians): {similarity}')
    query_embedding, search_results = kb.Semantics.search(
      query=query,
      corpus=corpus,
      similarity_bounds=similarity
    )
    if not search_results:
      logger.error(f'No Results Found for query: {query}')
      return 1
    
    # Extract the Results
    snippets: dict[str, list[str]] = {}
    for doc_name, doc_results in search_results.items():
      logger.debug(f'Document: {doc_name}')
      snippets[doc_name] = []
      for result in doc_results:
        logger.debug(f'Result Similarity (Radians): {result['metadata']["Similarity"]}')
        read_start = result['metadata']['DocChunks'][0]['start']
        read_chunk = result['metadata']['DocChunks'][0]['len']
        logger.debug(f'Reading Document {doc_name} from {read_start} to {read_start+read_chunk}')
        snippets[doc_name].append(''.join(corpus.read_document(doc_name, chunk=read_chunk, start=read_start, end=read_start+read_chunk)))

    logger.info(f'Found {sum(map(len, search_results.values()))} Results accross {len(search_results)} Documents')
    sys.stdout.write(json.dumps({
      'query': query,
      'results': snippets,
    }))
    return 0

def _parse_flag(flag: str) -> tuple[str, str]:
  assert flag.startswith('-')
  if '=' in flag: return flag.lstrip('-').split('=', maxsplit=1)
  else: return flag.lstrip('-'), True

def main(argv: list[str], env: dict[str, str]) -> int:
  logger.debug(f'ARGV: {argv}')
  args = deque(a for a in argv if not a.startswith('-'))
  logger.debug(f'Arguments: {args}')
  flags = dict(_parse_flag(f) for f in argv if f.startswith('-'))
  logger.debug(f'Flags: {flags}')
  subcmd = args.popleft()
  if subcmd.lower() == 'doc':
    action = args.popleft()
    if action.lower() == 'embed': return SubCommand.generate_doc_embeddings(
      corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
      document_name=args.popleft(),
      chunk_size=int(flags.get('chunk', '4096')),
      batch_size=int(flags.get('batch', '8'))
    )
    elif action.lower() == 'distill': return SubCommand.distill_doc(
      corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
      document_name=args.popleft(),
      chunk_size=int(flags.get('chunk', '4096')),
      batch_size=int(flags.get('batch', '8'))
    )
    elif action.lower() == 'summarize': return SubCommand.summarize_doc(
      corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
      document_name=args.popleft(),
    )
    else: raise RuntimeError(f'Unknown Action for {subcmd}: {action}')
  elif subcmd.lower() == 'corpus':
    action = args.popleft()
    if action.lower() == 'add': return SubCommand.add_document_to_corpus(
      corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
      doc_name=args.popleft(),
      doc_kind=args.popleft(),
      doc_src=(lambda a: 'fd:0' if a == '-' else a)(args.popleft()),
      doc_prefix=args.popleft() if args else None,
    )
    elif action.lower() == 'validate': return SubCommand.validate_corpus(
      corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
    )
    elif action.lower() == 'search':
      degrees = args.popleft()
      if ',' in degrees: degrees = tuple(map(float, degrees.split(',')))
      else: degrees = (-1 * float(degrees), float(degrees))
      query = args.popleft()
      if query == '-': query = sys.stdin.read()
      return SubCommand.search_corpus(
        corpus_root=pathlib.Path(flags.get('kb', './KnowledgeBase')),
        query=query,
        # similarity=tuple(map(lambda x: x * (math.pi / 180), degrees)), # Convert Degrees to Radians
        similarity=degrees,
      )
    else: raise RuntimeError(f'Unknown Action for {subcmd}: {action}')
  else: raise RuntimeError(f'Unknown Subcommand: {subcmd}')

if __name__ == '__main__':
  logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'), stream=sys.stderr)
  _rc = 2
  try:
    _rc = main(sys.argv[1:], os.environ)
  except SystemExit as e:
    _rc = e.code
  except Exception as e:
    logger.exception('Unhandled Error')
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    exit(_rc)
