# ADR-015: Open/Closed LLM Provider Registry Pattern

## Status
Accepted (Planned for `v1.3.0`, Phase 12.3)

## Context
Using monolithic procedural `if/elif` chains in `config.get_llm_model()` violates the Open/Closed Principle (OCP). Adding support for new inference engines (e.g. Anthropic Claude, Cohere, local Ollama) requires modifying core configuration source code and risks regression across existing providers.

## Decision
1. **Registry Pattern**: Implement a decoupled `PROVIDER_REGISTRY: dict[str, Callable]` mapping provider identifiers to lazy model factory functions.
2. **Pluggable Registration API**: Expose `config.register_provider(name: str, factory: Callable)` to allow runtime addition of custom or enterprise inference backends.
3. **Lifecycle Isolation**: Encapsulate `.env` loading inside an explicit `initialize_config()` function to prevent environment variable pollution during unit testing.

## Consequences
- **Positive**: Full compliance with OCP; allows zero-code modification when adding new model providers; isolates test suites from global environment mutation.
- **Negative**: Requires registering custom provider factories before invocation.
