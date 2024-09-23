"""

Generate Documentation for a Project using an Intelligent Agent.

> This SubModule is actively developed with the meta-goal of evaluating
> end to end usage of the `DevAgent`.

The Documentation Generation framework is:

- Articulate the Documentation you want to generate:
  - What are the major topics and/or themes.
  - Who is your audience?
  - What is the scope (breadth & depth) of the documentation?
  - What is your message?
  - What should key take aways be?
- Gather relavant sources & other information:
  - Are there specific documents, files or other artifacts to specifically document?
  - Is there contextual information or implicit knowledge relevant to your content?
  - What information or other semantics is relevant to the delivery of your message?
  - Is there a gap in your knowledge?
- Generate the Documentation:
  - Establish the hierarchical structure of the documentation (dependant on scope):
    - Is it a single document that is unstructured message?
    - Is it a single document w/ multiple sections & sub sections which warrant a ToC?
    - Is it multiple documents where each document follows varying formats or templates?
  - Establish the Writing Process (At the Document Level):
    - For well formed & structured Semantics:
      - Write in a high level skeleton of the document (ie. Section Headers & subheaders)
      - Iteratively fill in details per section.
      - Revise & expand on sections
      - Evaluate the entire document; should we continue to fill in details or should the document
        be re-organized, collapsing or breaking out topics?
    - For unstructured or otherwise "fuzzy" Semantics:
      - Write the content, prioritizing the most important information first.
      - Evaluate the content written; did you capture your message; did you capture your key takeaways?
  - Continuously Iterate as things change; introduce new topics, references, semantics and information.

"""
import json, logging
from typing import Any

logger = logging.getLogger(__name__)

class JSONCapture:

  def __init__(self, sink: str):
    self._sink_src = sink
    self._sink = open(sink, mode='a+')

  def __del__(self):
    if not self._sink.closed:
      self._sink.flush()
      self._sink.close()
  
  def __bool__(self): return not self._sink.closed

  def write(self, o: Any):
    try: self._sink.write(json.dumps(o) + '\n')
    except json.JSONDecodeError: logger.warning('Failed to Encode Capture Object; Not valid JSON')
  
  def close(self):
    if not self._sink.closed:
      self._sink.flush()
      self._sink.close()
