"""

# Task: Generate Documentation

An automated `Intelligent Agent` that generates the documenation's content.

Content Generation follows:

- Given the Document Vision & other informational Context, an LLM generates documentation

"""
from __future__ import annotations
from typing import IO

## Local Imports
from ... import llm, KnowledgeBase as kb
from . import DocCtx
##

ctx = DocCtx.Context()
"""The Shared Context"""

def entrypoint(
  doc_vision: str,
  doc_ref: str,
  extra_doc_ctx: str,
  corpus: kb.Corpus,
  capture: IO | None = None,
):
  """The Entrypoint of the Documentation Generation Task.

  A Document Vision & Reference Material must be provided.

  A Document Context will be generated; it will be supplemented with the provided doc_ctx
  
  A Capture Sink may be attached in which case the step by step "thought process" will be dumped to.
  """
  # Reset the Corpus State
  ctx.clear()
  ctx.capture = capture
  ctx.corpus = corpus

  if not doc_vision.strip(): raise ValueError('Empty Document Vision Provided')
  doc_msgs = [
    { 'role': 'assistant', 'content': 'I understand; provide me with a Documentation Vision to guide myself with.' },
    { 'role': 'user', 'content': doc_vision },
  ]

  if doc_ref.strip(): doc_msgs += [
    { 'role': 'assistant', 'content': 'I understand; provide me with reference material to consult.' },
    { 'role': 'user', 'content': doc_ref },
  ]

  ### Dynamically Generate the Context
  # TODO: Generate a Topic using an LLM
  topic = "This document outlines a comprehensive plan for creating and structuring documentation for an AI-powered developer productivity toolkit called DevAgent. It emphasizes the importance of clear, concise, and thorough documentation that caters to highly skilled software developers. The document describes the toolkit's purpose, target audience, core message, and key objectives, highlighting DevAgent's role in enhancing developer workflows through AI-assisted information management and knowledge acquisition. It also provides a framework for organizing the documentation, including sections on architecture, core modules, and usage guidelines. The plan emphasizes the dynamic and iterative nature of the documentation process, utilizing a directed graph approach for generating content, with the ultimate goal of enabling developers to effectively contribute to and extend the DevAgent package."
  ctx.grow(topic) # TODO: Specify the topics used to Grow the Context
  raise NotImplementedError
  ctx_lookup = ctx.lookup(...)
  doc_ctx: str = render_ctx(ctx_lookup) + extra_doc_ctx.strip() # TODO: Generate a Query to Render the Context
  if doc_ctx.strip(): doc_msgs += [
    { 'role': 'assistant', 'content': 'I understand; provide me with any relevant context.' },
    { 'role': 'user', 'content': doc_ctx },
  ]

  ### Assemble the Messages
  msgs = [
    { 'role': 'system', 'content': GEN_DOC_SYS_PROMPT }, # System Prompt
    { 'role': 'user', 'content': GEN_DOC_INSTRUCT }, # Instructions
    *doc_msgs,
    { 'role': 'assistant', 'content': 'I will a) use the provided Vision as my guide, b) consult the provided information and c) follow the instructions you previously provided.' }, # Recap
    { 'role': 'user', 'content': 'Proceed with writing the documentation.' }, # Prompt
  ]
  if capture: capture.write({
    'kind': 'LLM Documentation Generation Messages',
    'content': msgs,
  })

  ### Have the LLM Generate Documentation
  response = llm.chat(*msgs)
  if capture: capture.write({
    'kind': 'LLM Documentation Generation Response',
    'content': response,
  })
  return response

GEN_DOC_SYS_PROMPT = """
You are an expert technical writer whose goal is to write documentation as described by the shared Vision.
""".strip()
GEN_DOC_INSTRUCT = """
Here are instructions to follow when writing documentation: 
a) Strictly adhere to the Vision provided following the outlined purpose, objectives, target audience, core message & key takeaways.
b) Utilize the provided Reference Materials & the Context to inform your content. The Reference is some source material directly influencing the content. The Context provides relevant information to inform what you write.
c) Generate specific, relevant content addressing the needs of target Audience.
d) Maintain a clear, concise, and professional tone; speak plainly & prioritize reader comprehension.
e) Use appropriate Markdown formatting for effective structure.
f) Explain complex concepts clearly, breaking them down when necessary & including accurate snippets and/or examples to tie it all together.
g) Be salient in your explanations & structure your prose so the most pertinent information comes first.
h) State any limitations or insufficient information clearly.
""".replace('\n', ' ').strip()

### Helpers

def render_ctx(lookup_results: dict) -> str:
  """Render the Context relevant to the query"""
  return ''