# Designing Computational Models for System Models through Cognition

## Abstract

In this document, I aim to articulate how we can design computational systems that implement system models through cognition guided by frameworks.

[TOC]

## Frameworks, Systems & Models

An Intelligent Agent is an entity that has the ability to acquire & apply knowledge and skills, either intrinsically or through emulation.

A Framework is a basic conceptual structure that serves as a foundation for various processes, systems, or ideas. Frameworks guide an intelligent agent's cognition.

A System is an object or process that functions as a composite of discrete parts, principles or procedures cooperatively operating together. For our context, it is imperative that when work is applied to a system by an external entity, the system's internal state changes; the system may consequently produce some side-effect to it's environment (conceptualized as the system doing work).

A Model is a cognitive representation of a system, implemented as another system. It aims to reproduce or emulate the behavior, partially or in full, of the system being modeled.

A System Model is a cooperative collection of conceptual ideas & principles that serve as a vessel through which intelligent agents articulate and comprehend systems, physical, virtual or conceptual.

A Computational System is a collection of computer hardware executing software which produce a side effect or otherwise calculate information for further use. A Computational System may have higher order interfaces & concepts that are composites of their fundamental constituents.

As such, an intelligent agent can use computational systems to implement, evaluate & enact their system models.

## Design Framework

Software development is the loose design framework that encompasses:

- Articulating a conceptual model of a system.
- Instructing a computer how to operate in order to achieve a side-effect or produced information.

Generally this entails using a programming language to both articulate a conceptualization & implement computational instructions. Programming paradigms such as imperative, declarative, functional & object oriented, are all specialized applications of this framework. I propose another specialized application of the software development framework.

My framework focuses on delineation of the software development process into constituent procedures that isolate the cognitive representation of the various systems. It also provides principles to guide layering of these procedures to successfully achieve results.

First, the major constituent procedures are:

- Develop the system model; articulate the system you are modeling & the desired side-effects of the model when work is applied.
- Concurrently develop the software interface to the computational system or otherwise reuse an existing computational system.
- Develop a computational representation of the system model by mapping the system model onto the computational system.
  - If the system model cannot be mapped then iterate from the start of the framework.
  - Otherwise the system model is ready to be executed.

The general flow of these procedures follows:

```mermaid
flowchart TD;
  A(Start)--> B(Articulate & Develop system model) & C(Develop Computational System)--> D(Develop Representational Mapping)
  D -- evaluate --> ready([Can the system model be fully mapped to the computational system?])
  ready -. yes .-> E(Execute)
  ready -. no .-> A 
```

Next, let's discuss the principles that should guide designs & developments to achieve consistent & effective results. Not all principles need be applied every time, but collectively they should ensure success.

### The First Principle

The first principle describes ways to articulate & develop system models using `types` & `protocols`.

Plainly:

- Use `types` to describe the data or information that can have work applied on them or by them. Such work applied on or by types should induce or contribute in part or in full to a state change or a side-effect of the system.
- Use `protocols` to define similar functionality & context within a system to enable consistent behavior across `type` sets; functionality relates to actionability while context relates to state.

More formally:

- A `type` is an implementation of semantics & capabilities applied to data.

- A `protocol` declares a set of semantics & capabilities such that any `type` having these semantics & capabilities, irrespective of implementation, can be classified as enacting the `protocol`.
- Semantics deal with the meaning of data & how this meaning is contextualized within an intelligent agent's understanding of a system. Semantics include implied knowledge, logical rules & descriptive boundaries.
- Capabilities deal with the actions & abilities that can be performed on, by or with the data contextualized to its semantics within the system.

### The Second Principle

> TODO: Need to think through & expand on this more

The second principle describes a common approach to articulating a system model using a Domain Specific Language (DSL) based on Natural Language (NL) Paradigms.

Plainly:

- Using a DSL-NL simplifies the process of articulating the system model & ensures a common vernacular.

  - Natural language (NL) is a capability intelligent agents use to externalize knowledge; the process of articulating comprehension & implicit knowledge is already captured in the prose of natural language. However, in its native forms (such as English), its inherent design is generalized which gives rise to ambiguity & complexity.

  - Crafting a DSL-NL simplifies articulating system models with precision & accuracy. Likewise, development of the language itself serves as a process to refine understanding, narrow scope & materialize implicit knowledge for sharing & distribution.

  - The DSL-NL must be self describing to allow extension of the language to fit the domain.

- The DSL-NL should be fully deterministic to allow repeatable parsing while ensuring ambiguity is minimized (ideally eliminated). Ambiguity is a strong indicator that a system model is incomplete. If a concept cannot be plainly articulated then it is likely that a concept is not mappable to the computational system.
- The granularity & verbosity of as DSL-NL is a tradeoff between time & effort. The depth, precision & accuracy of the DSL-NL should minimally meet the requirements necessary to map to a computational system. These requirements are relative & subjective to the intelligent agents implementing the overall solution.
- Immediate or excessive generalization is an indicator that a system model is incomplete. Likewise struggling to structure concepts into a simple hierarchy can indicate a gap in implied knowledge. Generalization is neither encouraged nor discouraged but its use should be heavily scrutinized as to minimize ambiguity.

### The Third Principle

> TODO: Need to think through & expand on this more

The third principle details the scopes & strengths of each procedure & how they should be interweaved.

Plainly:

- The system model Procedure captures the "what" , "why" & "how" of systems qualifying the semantics & capabilities.
  - What constitutes the system & why?
  - How do the constituents interact with one another?
- The Computational System Procedure captures the "what" , "why" & "how" of computation: various paradigms & frameworks for efficiently instructing a computer.
  - What computational capabilities does a computer or a network of computers have? How are they applied?
  - How do we represent mathematical data structures (like a graph) into a computational framework? Why?
- The Representational Mapping Procedure captures the "how" of computing a system model. While no new types or protocols are introduced in this procedure, imperative work is introduced to coerce system model `types` into a composite of computational `types`.
  - A Basic example could be pattern extraction. A system model could imply `match the pattern foo from the input` while a computational system would deal with the iteration & indexing of a character buffer, the loading & management of the file's content into memory & how pattern matching is conducted.
  - Changes in the evolution of computational capabilities & modeled systems reflect in the differential between commits of the Representational Mapping source code. This allows for traceability in changing paradigms, implicit knowledge & growing capabilities.
  - An inability or high difficulty in implementing the imperative work to map from system space to computational space quickly makes it apparent that there is a gap in some component in the overall solution.
  - Computational optimization is decoupled from the system models allowing for faster refactoring to leverage expanded computational capabilities. When the computational system is tightly integrated w/ the system model (such as in traditional programming frameworks), then major refactoring is required to optimize unless a highly skilled intelligent agent developed the solution with this in mind; I assert the later is an exception & not the case.
