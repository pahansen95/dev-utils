"""

The KnowledgeBase Module provides the following Features:

- Document Corpus: Manage, Organize & Analyze a collection of Documents
- Semantic Search: Search the Corpus for Information based on Semantic Similarity

"""
from __future__ import annotations
import pathlib, yaml, functools, time, itertools, logging, shutil, json, unicodedata, re, hashlib, math, struct
from typing import Literal, TypedDict, Optional, Any
from collections.abc import Iterable, Iterator, ByteString
from dataclasses import dataclass

# raise NotImplementedError('Refactor to use the new NatLang Utils')
from ..Utils.NatLang import embed, chat

logger = logging.getLogger(__name__)

@dataclass
class Corpus:
  """The Collection of Documents"""
  root: pathlib.Path
  """The Root of the Corpuse Tree"""
  documents: dict[str, Document]
  """The set of documents in the Corpus, keyed on name"""

  @classmethod
  def factory(cls, root: pathlib.Path) -> Corpus:
    if isinstance(root, pathlib.Path):
      if not (root.exists() and root.resolve().is_dir()): raise FileNotFoundError(root)
      logger.debug('Assembling Documents for Corpus')
      docs = {}
      for full_path in Corpus.find_docs(root):
        doc = Document(
          root=full_path.relative_to(root),
          metadata=Document.load_metadata_from_disk(full_path),
          semantics=Document.load_semantics_from_disk(full_path),
        )
        docs[doc.metadata['name']] = doc
      return cls(root=root, documents=docs)
    else: raise TypeError(type(src))

  @staticmethod
  def find_docs(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Find the Documents in the Corpus returning their full path"""
    yield from (p.parent for p in filter(
      lambda p: p.read_text().strip() == Document.__name__,
      root.rglob('**/.kind')
    ))
  
  @staticmethod
  def init_document(corpus_root: pathlib.Path, doc: Document):
    logger.debug(f'Initializing the Document: {doc.metadata["name"]}')
    assert corpus_root.exists()
    (doc_root := corpus_root / doc.root).mkdir(mode=0o755, exist_ok=True, parents=True)
    (kind_file := doc_root / '.kind').touch(mode=0o755, exist_ok=True)
    kind_file.write_text(Document.__name__)

  @staticmethod
  def save_document_data(corpus_root: pathlib.Path, doc: Document):
    logger.debug(f'Saving the Document: {doc.metadata["name"]}')
    assert corpus_root.exists()
    if not (doc_root := corpus_root / doc.root).exists(): raise RuntimeError('Document Not Initialized')
    # Write the Metadata & Semantics
    Document.save_metadata_to_disk(doc.metadata, doc_root)
    if doc.semantics: Document.save_semantics_to_disk(doc.semantics, doc_root) # Semantics might not be computed yet
    # Document Content is Immutable b/c a change in content invalidates all pre-cached semantics
  
  @staticmethod
  def load_document_data(corpus_root: pathlib.Path, doc: Document) -> tuple[Document.Metadata, Document.Semantics]:
    logger.debug(f'Loading the Document: {doc.metadata["name"]}')
    assert corpus_root.exists()
    if not (doc_root := corpus_root / doc.root).exists(): raise RuntimeError('Document Not Initialized')
    return (
      Document.load_metadata_from_disk(doc_root),
      Document.load_semantics_from_disk(doc_root),
    )

  def create_document(
    self,
    name: str,
    kind: Literal['text'],
    data: Iterator[str] | ByteString,
    prefix: pathlib.Path | None = None,
  ) -> Document:
    """Create a new Document adding it to the Corpus"""
    logger.debug(f'Document {name}: Create as {kind}')

    if name in self.documents: raise ValueError(f'Document {name}: already exists in the Corpus')

    doc_meta: Document.Metadata = {
      'created_at': int(time.time()),
      'document_kind': kind,
      'name': name,
      'publisher': {}, # TODO
    }
    assert prefix is None or not prefix.is_absolute()
    if prefix: doc_root = prefix / name
    else: doc_root = pathlib.Path(name.lstrip('/'))
    self.documents[name] = (doc := Document(
      root=self.root / doc_root,
      metadata=doc_meta,
      semantics={}, # No Semantics Generated
    ))

    Corpus.init_document(self.root, doc)
    Corpus.save_document_data(self.root, doc)
    logger.debug(f'Document{name}: Writing Content')
    if isinstance(data, ByteString): doc_size = doc.dump(self.root, data)
    elif isinstance(data, Iterator): doc_size = doc.write(self.root, data)
    else: raise TypeError(type(data))
    logger.debug(f'Document {name}: Byte Size: {doc_size}')

    return doc
  
  def update_document(self, name: str) -> Document:
    """Update the Document in the Corpus"""
    logger.debug(f'Document {name}: Update')
    doc = self.documents[name]
    logger.debug('Syncing Document')
    Corpus.save_document_data(self.root, doc)
    # TODO: Update Document Content
    return doc

  def delete_document(self, name: str) -> Document:
    """Delete the Document from the Corpus"""
    logger.debug(f'Document {name}: Delete')
    doc = self.documents[name]
    doc_root = self.root / doc.root
    if doc_root == self.root: raise RuntimeError('Bad Document Root Path: Path is Corpus Root')
    if doc_root.exists(): shutil.rmtree(doc_root)
    # TODO: Remove all parents if they are empty directories
    return doc
  
  def read_document(self, name: str, **kwargs) -> Iterator[str]:
    """Iteravely Read a Document's Content from disk"""
    return self.documents[name].read(self.root.resolve(), **kwargs)

  def slurp_document(self, name: str, **kwargs) -> str:
    """Slurp a Document's content into memory"""
    return self.documents[name].slurp(self.root.resolve(), **kwargs)
  
  def write_document(self, name: str, data: Iterator[Any]) -> int:
    """Iteravely Write Content to a Document; NOTE that Document Content is immutable & should only be written to during creation"""
    return self.documents[name].write(self.root.resolve(), data)
  
  def dump_document(self, name: str, data: Any) -> int:
    """Dump the Content to a Document; optimized when all data fits in memory; NOTE that Document Content is immutable & should only be written to during creation"""
    return self.documents[name].dump(self.root.resolve(), data)

@dataclass
class Document:
  """A Document in the Corpus"""
  root: pathlib.Path
  """The Root of the Document Tree relative to the Corpus Root; Must already exist"""
  metadata: Document.Metadata
  """Metadata about the document"""
  semantics: Document.Semantics
  """The Pre-Computed Semantics about the document"""

  class Metadata(TypedDict):
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

    @staticmethod
    def marshal(data: Document.Metadata) -> bytes:
      """Marshal the Metdata to a Byte String"""
      return yaml.safe_dump({'kind': Document.Metadata.__name__, 'spec': data}, sort_keys=False)

    @staticmethod
    def unmarshal(data: bytes) -> Document.Metadata:
      """Unmarshal the Metadata from a Byte String"""
      body = yaml.safe_load(data)
      if body['kind'] != Document.Metadata.__name__: raise ValueError('Not a Document Metadata Object')
      return body['spec']
  
  class Semantics(TypedDict):
    """The Computed Semantics for a document"""
    summary: Optional[str]
    """A Salient Summary of the Document"""
    # classifications: Optional[...]
    # """Classifiers grouping this document relative to the Corpus"""
    distillate: Optional[list[str]]
    """The Set of Salient Information extracted from the Document"""
    embeddings: Optional[list[embed.Embedding]]
    """Vector Embeddings computed for the Document"""

    # @staticmethod
    # def marshal(data: Document.Semantics) -> bytes:
    #   """Marshal the Semantics Data to a Byte String"""
    #   return yaml.safe_dump({'kind': Semantics.__name__, 'spec': data})

    # @staticmethod
    # def unmarshal(data: bytes) -> Document.Semantics:
    #   """Unmarshal the Semantics Data from a Byte String"""
    #   body = yaml.safe_load(data)
    #   if body['kind'] != Semantics.__name__: raise ValueError('Not a Semantics Object')
    #   return body['spec']
  
  def content_size(self, parent: pathlib.Path) -> int:
    """Get the Size of the Document's Content in Bytes; must provide an absolute path to the parent"""
    assert parent.is_absolute() and parent.exists()
    if self.metadata['document_kind'] == 'text': return self._text_size(parent)
    else: raise NotImplementedError(f"Unsupported Document Kind: {self.metadata['document_kind']}")

  def _text_size(self, parent: pathlib.Path) -> int:
    return (parent / self.root / 'content/blob.txt').stat().st_size

  def read(self, parent: pathlib.Path, start: int = 0, end: int | None = None, chunk: int = 65_536) -> Iterator[str]:
    """Iteratively read the Document from disk. If end is None, read to the end of the document"""
    assert parent.exists()
    if not (parent / self.root / 'content').exists(): raise ValueError('Document has no content')
    if end is None: end = self.content_size(parent)
    assert end <= self.content_size(parent), f'Want {end} but max is {self.content_size(parent)}'
    logger.debug(f'Reading `{self.metadata["document_kind"]}` Document: {self.metadata["name"]}, Start: {start}, End: {end}, Chunk: {chunk}')
    if self.metadata['document_kind'] == 'text': return self._read_text(parent, start, end, chunk)
    else: raise NotImplementedError(f"Unsupported Document Kind: {self.metadata['document_kind']}")
  
  def _read_text(self, parent: pathlib.Path, start: int, end: int, chunk: int) -> Iterator[str]:
    assert parent.exists()
    with (parent / self.root / 'content/blob.txt').open('r') as f:
      logger.debug(f'{start=}')
      f.seek(start)
      idx = 0
      while (pos := f.tell()) < end:
        logger.debug(f'{idx=}: {pos=}, {end=}')
        yield f.read(min(chunk, end - pos))
        idx += 1
  
  def slurp(self, parent: pathlib.Path) -> bytes:
    """Read the entire Document into memory"""
    assert parent.exists()
    if self.metadata['document_kind'] == 'text': return (parent / self.root / 'content/blob.txt').read_bytes()
    else: raise NotImplementedError(f"Unsupported Document Kind: {self.metadata['document_kind']}")
  
  def write(self, parent: pathlib.Path, data: Iterator[Any]) -> int:
    """Iteratively write the Document Context to disk returning the total bytes written"""
    assert parent.exists()
    (parent / self.root / 'content').mkdir(mode=0o755, exist_ok=True)
    if self.metadata['document_kind'] == 'text': return self._write_text(parent, data)
    else: raise NotImplementedError(f"Unsupported Document Kind: {self.metadata['document_kind']}")
  
  def _write_text(self, parent: pathlib.Path, src: Iterator[str]) -> int:
    """Write the text to disk; return the total bytes written"""
    assert parent.exists()
    assert isinstance(src, Iterator)
    with (self.root / 'content/blob.txt').open(mode='w+') as f:
      return sum(map(f.write, src))
  
  def dump(self, parent: pathlib.Path, data: ByteString):
    """Dumps the Source onto disk; optimizes writes when the document fits in memory"""
    assert parent.exists()
    (parent / self.root / 'content').mkdir(mode=0o755, exist_ok=True)
    if self.metadata['document_kind'] == 'text': return (parent / self.root / 'content/blob.txt').write_bytes(data)
    else: raise NotImplementedError(f"Unsupported Document Kind: {self.metadata['document_kind']}")
  
  @staticmethod
  def save_semantics_to_disk(semantics: Document.Semantics, doc_root: pathlib.Path):
    """Save the Semantics Data to Disk; must provide an absolute paht"""
    ### Make sure Things exist first
    assert doc_root.is_absolute()
    if not doc_root.exists(): raise FileNotFoundError(doc_root)
    (semantics_dir := doc_root / 'semantics').mkdir(mode=0o755, exist_ok=True)
    ###

    ### Sync Summary
    if semantics.get('summary', ''):
      (summary_file := semantics_dir / 'summary.txt').write_text(
        semantics['summary']
      )
    ### Sync Distillate
    if semantics.get('distillate', []):
      (distillate_file := semantics_dir / 'distillate.yaml').write_text(
        yaml.safe_dump(semantics['distillate'], sort_keys=False)
      )
    ### Sync Semantics
    assert isinstance(semantics, dict)
    if semantics.get('embeddings', []):
      assert isinstance(semantics["embeddings"], list), semantics["embeddings"]
      assert not isinstance(semantics["embeddings"], dict), semantics["embeddings"]
      (embeds_dir := semantics_dir / 'embeds').mkdir(mode=0o755, exist_ok=True)        

      new_embeds: set[str] = set()
      old_embeds: set[str] = set(f.name.rsplit('.', maxsplit=1)[0] for f in embeds_dir.glob('*.bin'))
      # logger.debug(f'Found Existing Embeddings: {sorted(old_embeds)}')
      for e in semantics["embeddings"]:
        assert isinstance(e, dict), e
        fingerprint = hashlib.md5(e['buffer'], usedforsecurity=False).hexdigest()
        new_embeds.add(fingerprint)
        if fingerprint in old_embeds: continue # Skip Writing existing Embeddings
        # otherwise write the new embedding
        if fingerprint not in old_embeds: (embeds_dir / f'{fingerprint}.bin').write_bytes(e['buffer'])
        (embeds_dir / f'{fingerprint}.spec.yaml').write_bytes(
          embed.Embedding.marshal(e)
        )
      # Remove any old Embeddings that aren't carried through in the new embeddings
      assert new_embeds
      # logger.debug(f'Added New Embeddings: {sorted(new_embeds)}')
      del_embeds = old_embeds - new_embeds
      assert del_embeds.isdisjoint(new_embeds)
      # logger.debug(f'Removing Old Embeddings: {sorted(del_embeds)}')
      for fingerprint in del_embeds:
        (embeds_dir / f'{fingerprint}.bin').unlink()
        (embeds_dir / f'{fingerprint}.spec.yaml').unlink()
  
  @staticmethod
  def load_semantics_from_disk(doc_root: pathlib.Path) -> Document.Semantics:
    """Load the Semantics from disk; must provide an absolute path."""
    assert doc_root.is_absolute()
    ### Make sure Things exist first
    if not doc_root.exists(): raise FileNotFoundError(doc_root)
    ###
    semantics = {}
    if not (semantics_dir := doc_root / 'semantics').exists(): return semantics # Short-Circuit if no semantics exist
    if (summary_file := semantics_dir / 'summary.txt').exists():
      semantics['summary'] = summary_file.read_text()
    
    if (distillate_file := semantics_dir / 'distillate.yaml').exists():
      semantics['distillate'] = yaml.safe_load(distillate_file.read_bytes())
    
    if (embeds_dir := semantics_dir / 'embeds').exists():
      semantics['embeddings'] = list(map(
        lambda f: embed.Embedding.unmarshal(
          f.with_suffix('.spec.yaml').read_bytes(),
          f.read_bytes(),
        ),
        embeds_dir.glob('*.bin'),
      ))
    
    return semantics

  @staticmethod
  def save_metadata_to_disk(metadata: Document.Metadata, doc_root: pathlib.Path):
    """Write the Metadata to Disk; must provide an absolute path"""
    assert doc_root.is_absolute()
    ### Make sure Things exist first
    if not doc_root.exists(): raise FileNotFoundError(doc_root)
    (metadata_dir := doc_root / 'metadata').mkdir(mode=0o755, exist_ok=True)
    ###
    (metadata_dir / 'spec.yaml').write_text(
      yaml.safe_dump(metadata, sort_keys=False)
    )
  
  @staticmethod
  def load_metadata_from_disk(doc_root: pathlib.Path) -> Document.Metadata:
    """Load the Metadata from Disk; must provide an absolute path"""
    assert doc_root.is_absolute()
    ### Make sure Things exist first
    if not doc_root.exists(): raise FileNotFoundError(doc_root)
    ###
    if not (
      (metadata_dir := doc_root / 'metadata').exists()
      and (metadata_file := metadata_dir / 'spec.yaml')
    ): raise RuntimeError('Malformed Document: Missing Metadata')
    return yaml.safe_load(metadata_file.read_bytes())

class Semantics:
  """TODO

  - Semantic Chunking:
    > NOTE: Probably needs a preclassifier to determine if the approach matches the content
    - Split a Document on "sentences"
    - embed each sentence
    - connect the dots
    - walk the path (aka narrative)
    - establish node groups based on distances
    - each group becomes a document chunk
  - Sliding Window Chunking:
    - Split on N Chars
  
  """

  @staticmethod
  def embed(*chunk: str) -> embed.Embedding:
    """Generate an Embedding for each chunk of text"""
    return Embeddings.embed(*chunk)
  
  @staticmethod
  def distill(*chunk: str, existing: list[str] = []) -> list[str]:
    """Extract a list of Salient Information from the provided chunks merging with the existing list"""
    msg_chain: list[chat.Message] = [
      {
        'role': 'system',
        'content': 'You are an automated text processor. Given a chunk of text, you extract salient information.'
      },
      *([
        {
          'role': 'assistant',
          'content': 'Please provide the text chunk; I will extract the salient information. I will respond with a JSON Object having a single key `items` containing the extracted information as a list of strings.'
        }
      ] if not existing else [
        {
          'role': 'user',
          'content': f"""
I am providing previously extracted salient information:

{'\n'.join(f'- {item}' for item in existing)}
""".strip()
        },
        {
          'role': 'assistant',
          'content': 'Please provide the text chunk; I will extract only new salient information not already provided. I will respond with a JSON Object having a single key `items` containing the extracted information as a list of strings.'
        }
      ]),
    ]
    # TODO: Make calls concurrently & Retry on Malformed Responses
    salient: set[str] = set(existing)
    for c in chunk:
      attempt_count = 0
      while attempt_count < 5:
        resp: str = chat.chat(*msg_chain, { 'role': 'user', 'content': c.strip() }).strip()
        attempt_count += 1
        try: resp_json: str = chat.extract_markdown_code_block(resp, 'json')
        except:
          logger.debug(f"Failed to Extract JSON Code Block from Response")
          resp_json = resp
        ### Normalize the Salient Information
        try:
          salient.update(json.loads(resp_json)["items"])
          break
        except:
          logger.warning(f"Malformed JSON Format: {resp}")
          continue
      if attempt_count >= 5: logger.error(f"Skipping Chunk Extraction due to too many failures: {c}")
    
    return [s for s in salient if s and s not in existing]

  # @staticmethod
  # def classify(*chunk: str) -> Iterable[str]:
  #   """Classify each chunk of text"""
  #   raise NotImplementedError

  @staticmethod
  def summarize(*items: str) -> str:
    """Summarize the list of text items"""
    return llm.chat(
      {
        'role': 'system',
        'content': 'You are a technical writer. Given a document, you will generate a salient summary of the contained semantics.'
      },
      {
        'role': 'assistant',
        'content': '''
Please provide the document, I will then generate a salient summary.
The summary will start by identify the general subject of the document & include a descriptive verb describing the subject.
Then, the summary will aim to describe the semantic content in paragraph form.
Each paragraph will be a salient summary of a major semantic theme. 
Each paragraph will start with the theme's subject & a descriptive verb.
The summary will consist of as many paragraphs as there are major semantic themes.
My response will not include any special formatting.
'''.replace('\n', ' ').strip()
      },
      {
        'role': 'user',
        'content': ' '.join(items)
      },
    )

  @staticmethod
  def search(
    query: str,
    corpus: Corpus,
    similarity_bounds: tuple[float, float],
  ) -> tuple[embed.Embedding, dict[str, embed.Embedding]]:
    """Search a Corpus for Semantic Information related to a query"""
    logger.debug(f"{similarity_bounds=}")
    if not (
      similarity_bounds[0] >= -1
      and similarity_bounds[1] <= 1
      and similarity_bounds[0] <= similarity_bounds[1]
    ): raise ValueError(f'Invalid Similarity Bounds: Expected a range within [-1, 1]')

    ### Generate an embedding for the Query & Pre-Compute Some of the DeSerializing Bits

    query_embed = embed.INTERFACE.embed(query)
    assert query_embed['shape'][0] == 1 # Batch Size 1
    bytesize, f = embed._VEC_DTYPE[query_embed['dtype']]
    vec_size, err = divmod(len(query_embed['buffer']), bytesize)
    vec_bytesize = len(query_embed['buffer'])
    if err: raise RuntimeError(f"Query Embedding declares a data type of {query_embed['dtype']} but it's size is not a multiple of {bytesize}")
    vec_fmt = f'{vec_size}{f}'.encode() # N Floats
    logger.debug(f'{vec_size=}, {vec_bytesize=}, {bytesize=}, {vec_fmt=}')
    query_vec: embed._vector_t = struct.unpack(vec_fmt, query_embed['buffer'])
    _vec_magn = sum(x**2 for x in query_vec)
    assert math.isclose(1.0, _vec_magn, rel_tol=1e6), f'Embedding is not a Unit Vector: {_vec_magn}'

    ### "Search the Corpus" for similiar embeddings within the threshold

    """A NOTE on "Searching the Corpus"
    
    - Different Embedding Model's have different dimensional represetnations so we can only compare against embeddings of the same model.
    - We assume all embeddings are already normalized to distance of 1, so we can just use dot product to make the comparison.

    A Naive implementation of a search is just to compute the dot product for every single embedding in the corpus.
    This won't scale but should be fine to get us started.
    Right now, the threshold is range of values to include; dot product will be in the range [-1, 1].
    
    A NOTE on calculating the Dot Product

    This is just a "test it" implemenation; it's going to be really freaking slow.
    We can calculate the dot product b/c these are unit vectors.
    The Dot Product is the sum of the pairwise product of each vector component
    Our Vectors are buffers, so we need to convert from their datatype first.

    """
    matches: dict[str, embed.Embedding] = {}
    """The Matching Embeddings, grouped by Document"""
    for name, doc in corpus.documents.items():
      ### Collect all Matching embeddings for the Doc
      matching_embeds: list[ByteString] = []
      matching_embeds_metadata = {
        'DocChunks': [],
        'Similarity': []
      }
      for embedding_batch in doc.semantics['embeddings']:
        assert embedding_batch['model'] == query_embed['model']
        assert embedding_batch['shape'][1] == query_embed['shape'][1]
        assert embedding_batch['dtype'] == query_embed['dtype']
        batch_size = embedding_batch['shape'][0]
        assert batch_size > 0
        logger.debug(f'{batch_size=}, {vec_size=}, {bytesize=}')
        assert batch_size * vec_size * bytesize == len(embedding_batch['buffer']), f'Expected {batch_size * vec_size * bytesize} bytes, got {len(embedding_batch["buffer"])}'
        embed_vec_batch = memoryview(embedding_batch['buffer'])
        for i in range(0, len(embed_vec_batch), vec_bytesize):
          # chunk_vector = _batch_embedding[i:i+chunk_bytes]
          # assert len(chunk_vector) == chunk_bytes
          # angle = Semantics._dotproduct(query_embed['buffer'], chunk_vector, bytesize, fmt)
          # chunk_vec = Semantics._unpack_vector(vec_fmt, _batch_embedding[i:i+vec_bytesize])
          ebmed_vec: embed._vector_t = struct.unpack(vec_fmt, embed_vec_batch[i:i+vec_bytesize])
          angle = math.sumprod(query_vec, ebmed_vec) # Dot Product
          logger.debug(f'Similarity Score: {angle}')
          assert angle >= -1 and angle <= 1
          # Short Circuit if angle out of bounds
          if not (angle >= similarity_bounds[0] and angle <= similarity_bounds[1]): continue
          matching_embeds.append(embed_vec_batch[i:i+vec_bytesize])
          matching_embeds_metadata['DocChunks'].append(embedding_batch['metadata']['DocChunks'][i // vec_bytesize])
          matching_embeds_metadata['Similarity'].append(angle)
          ### NOTE: OLD Implementation
          # matches[name].append(embed.Embedding.factory(
          #   _batch_embedding,
          #   shape=tuple(embedding_batch['shape'][1:]),
          #   dtype=embedding_batch['dtype'],
          #   model=embedding_batch['model'],
          #   metadata={
          #     'DocChunks': [ embedding_batch['metadata']['DocChunks'][i // chunk_bytes] ],
          #     'Similarity': angle,
          #   }
          # ))
        
      if matching_embeds: matches[name] = embed.Embedding.factory(
        *matching_embeds, # Will be concatenated
        shape=(vec_size,), # Just the shape of a single Vector
        dtype=query_embed['dtype'],
        model=query_embed['model'],
        metadata=matching_embeds_metadata,
      )

    ### Return the results; TODO: What do we return exactly?
    return (
      query_embed,
      matches
    )

  @staticmethod
  def embedding_content(
    corpus: Corpus,
    doc: str,
    *embeds: embed.Embedding,
  ) -> Iterator[tuple[tuple[int, int], str]]:
    """Iterate over the embedding (batches) for a document returning the document's corresponding content"""
    if doc not in corpus.documents: raise ValueError(f'Uknown Corpus Document: {doc}')
    for embed_idx, embed in enumerate(embeds):
      for batch_idx, batch in enumerate(embed['metadata']['DocChunks']):
        yield (
          (embed_idx, batch_idx),
          ''.join(corpus.read_document(
            doc,
            start=batch['start'],
            end=batch['start'] + batch['len'] - 1,
          ))
        )
