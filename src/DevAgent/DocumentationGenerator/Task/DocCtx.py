"""

# Documentation Context

The proposed implementation creates a dynamic context generation system that starts with a given query and builds a
network of related information. This network, or context graph, begins with the core topic and expands outward, gathering
increasingly tangential but potentially relevant information.

Each piece of information in this network is assigned a relevance score, indicating how closely it relates to the original
query. As the system explores further from the core topic, the relevance of new information naturally decreases, mimicking
how context becomes less directly relevant as it becomes more removed from the main subject.

The system uses the existing Knowledge Base and its Semantic Search capability to find related information. It explores
the context in a breadth-first manner, looking at directly related information before moving to more tangentially related topics.

There's a built-in mechanism to prevent the context from growing too large or including irrelevant information.
This is achieved through a combination of depth limits (how far from the original topic we're willing to explore)
and relevance thresholds (how loosely related we allow information to be before we stop including it).

Once this context network is built, it's used to inform the documentation generation process. The context provides
the AI with a rich background of related information, allowing it to generate more comprehensive and accurate documentation.

This approach is designed to be flexible and iterative. It starts with a simple implementation that can be easily
modified and expanded as we learn more about what kinds of context are most useful for documentation generation.

"""
from __future__ import annotations
import networkx as nx, functools, math, logging
from typing import IO, Literal
from dataclasses import dataclass, field, KW_ONLY
from collections import OrderedDict

from ... import KnowledgeBase as kb

logger = logging.getLogger(__name__)

@dataclass
class Context:
  """Context is an in-memory Knowledge Graph (KG) of supplementary information sourced from an On-Disk Knowledge Base (KB).

  The KG contains a set of Source Topics which serve as frames of reference within the larger semantic space.
  When a Source Topic is added to the context, the KB is searched for semantically relevant information; if any is found,
  it is added to the KG & an edge is created between the Topic & Related Information.

  ### TODO

  - Search the KB for Nth Order relevant information; For example:
    Topic
      Relevant Order 1 Item A
      Relevant Order 1 Item B
      Relevant Order 1 Item C
        Relevant Order 2 Item A
          Relevant Order 3 Item A
            Relevant Order 4 Item A
              etc...
                Relevant Order N Item A      
  
  """
  kg: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
  """The Knowledge Graph; Edges originate from SOURCE Nodes"""
  corpus: kb.Corpus | None = None
  """The Knowledge Base Corpus"""
  capture: IO | None = None
  """Optional Capture Sink"""

  def relevance_threshold(self, depth: int) -> float:
    return 0.0 # TODO: Implement; for now just allow everything

  def grow(self,
    topic: str,
  ):
    """Grow the Context with Information relevant to the topic.
    
    The Topic should be some semantically rich description that
    contextualizes use & frames itself relative to world knowledge.
    Avoid using single words & ambiguous sentences. Articulation
    & salience are key to high quality topics.
    
    """
    if self.corpus is None: raise RuntimeError('Must set the Context\'s Corpus First')

    ### Add the topic to the Context as a SOURCE Node

    ### TODO: These need to be customizable
    depth: int = 1
    relevance_threshold = self.relevance_threshold(depth=depth)
    ###

    ### Get the Root Source for this topic
    if topic in self.kg: raise NotImplementedError
    logger.debug(f'Growing Context({id(self)}) w/ Topic...\n{topic}')
    if self.capture: self.capture.write({
      'kind': 'Semantic Search',
      'content': f'Growing Context({id(self)}) w/ Topic...\n{topic}'
    })
    topic_embedding = kb.Semantics.embed(topic)
    """The Topic Embedding used for calculating relevancy"""
    _topic_vecs = kb.embed.Embedding.unpack(topic_embedding)
    assert len(_topic_vecs) == 1
    topic_vec = _topic_vecs[0]
    self.kg.add_node(
      topic, # TODO: For now, just use the string as the node since it's hashable
      kind='TOPIC',
      embedding=topic_embedding,
    )
    
    ### Generate Queries from the Topic
    queries = [ topic ] # TODO: Use an LLM to generate multiple queries for a single topic

    ### For Each Query...
    for query in queries:

      ### Search the KB w/ the Query
      _, search_results = kb.Semantics.search(
        query, self.corpus,
        similarity_bounds=(0.2, 1.0), # TODO: What should these values be
      )
      if len(search_results) < 1: # Short Circuit
        logger.warning('No Relevant Information for the topic could be found in the Knowledge Base')
        return

      ### For Each Result...
      for doc, embedding in search_results.items():
        logger.debug(f'Found related content to Topic in Document {doc}')
        embedding_vec = kb.embed.Embedding.unpack(embedding)
        ### TODO: Refactor
        assert len(embedding['shape']) == 2, embedding['shape']
        _, _vec_dims = embedding['shape']
        _vec_bytesize = kb.embed._VEC_DTYPE[embedding['dtype']].bytesize * _vec_dims
        ###
        for (embed_idx, batch_idx), content in kb.Semantics.embedding_content(self.corpus, doc, embedding):
          assert embed_idx == 0, embed_idx

          ### Calculate the relevancy relative to the Source Topic
          # Calculate the Dot Product between the Search Result Embedding & the Topic's Embedding
          relevance: float = math.sumprod(topic_vec, embedding_vec[batch_idx])
          logger.debug(f'Relevance Score: {relevance}')

          ### Add the search results to the Context pruning any results that fall below a relevancy decay threshold
          if relevance >= relevance_threshold:
            logger.debug(f'Related Content Chunk={embed_idx}, Batch={batch_idx}')
            _start = _vec_bytesize * batch_idx
            _stop = _start + _vec_bytesize
            assert _start >= 0 and _start < len(embedding['buffer']), f"{_start=}, {len(embedding['buffer'])=}"
            assert _stop > 0 and _stop <= len(embedding['buffer']), f"{_stop=}, {len(embedding['buffer'])=}"
            if content in self.kg: raise NotImplementedError
            if self.capture: self.capture.write({
              'kind': 'Context Management',
              'content': f'Adding Relevant Content from `{doc}({_start}:{_stop})` to Context({id(self)})...\n\n' + content
            })
            self.kg.add_node( # Add the Related Content Node
              content,
              kind='RELATED',
              embedding=kb.embed.Embedding.factory(
                embedding['buffer'][_start:_stop], # Slice out the individual Embedding Vector from the batch
                shape=tuple(embedding['shape'][1:]),
                dtype=embedding['dtype'],
                model=embedding['model'],
                metadata={
                  'DocChunks': [ embedding['metadata']['DocChunks'][batch_idx] ],
                  'Similarity': [ embedding['metadata']['Similarity'][batch_idx] ]
                }
              )
            )
            self.kg.add_edge(
              topic,
              content,
              'TOPIC->RELATED',
              depth=depth,
              relevance=relevance
            )

    ### TODO: For each added CONTEXT Node, continue searching for related context; updating the Edge Depth
  
  def lookup(self,
    query: str,
    relevancy: float = 0.0,
  ) -> dict[str, list[dict[str, str | float]]]:
    """Lookup relevant information to the query"""
    ### Determine the SOURCE Nodes most relevant to the Query

    ### For each SOURCE Node, traverse the Context Tree

      ### Traverse the Entire Breadth

      ### Prune Traversal Branches below the relevancy score (by default include all results)
    
    ### Return the Results

    return {
      topic: [
        {
          'content': ...,
          'score': ...
        } for result in [ ... ]
      ] for topic in [ ... ]
    }

  def clear(self):
    """Clears all data in the knowledge graph; effectively resetting it"""
    self.kg.clear()
