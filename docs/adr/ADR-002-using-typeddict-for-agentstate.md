# ADR-002: Using `TypedDict` for `AgentState` Over Pydantic `BaseModel`

## Status
Accepted (Implemented in `v0.7.0`, Refined in `v0.8.4`)

## Context
LangGraph state machines require state objects passed across graph nodes and checkpointed into state memory (`MemorySaver`). Python developers often choose Pydantic `BaseModel` for automatic validation. However, strict Pydantic serialization adds validation overhead during graph state transitions and can raise runtime state serialization exceptions when custom non-serializable objects (such as LLM client instances) are temporarily stored.

## Decision
Use a standard Python `TypedDict` for the core `AgentState` schema in `agent/state.py`. Enforce payload contracts using Pydantic V2 JSON models exclusively at LLM tool boundary outputs (`JobRequirementsOutput`, `DeepScreenOutput`).

## Consequences
- **Positive**: Eliminates state checkpointer serialization overhead; matches official LangGraph idiomatic state management patterns; provides clean, native dictionary access across all graph nodes.
- **Negative**: Type checking relies on static type checkers (`mypy`, `pyright`) rather than runtime validation during state assignment.
