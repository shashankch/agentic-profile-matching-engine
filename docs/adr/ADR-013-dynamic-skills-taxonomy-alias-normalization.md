# ADR-013: Configuration-Driven Skills Taxonomy & Semantic Alias Normalization

## Status
Accepted (Planned for `v1.2.0`, Phase 11.3)

## Context
Hardcoding skill strings directly in source code classes (`MetadataExtractor.SKILLS = [...]`) causes silent false negatives for unlisted technologies (e.g. `ArgoCD`, `Pulumi`, `dbt`, `LlamaIndex`), fails on casing variations, and requires code deployments and pipeline rebuilds to update the recruitment domain vocabulary.

## Decision
1. **External Taxonomy Configuration**: Externalize skills and technology domains into `config/skills_taxonomy.yaml` with schema validation.
2. **Semantic Alias Mapping**: Implement alias and stemming normalization mappings (e.g. `"K8s"` $\to$ `"Kubernetes"`, `"Golang"` $\to$ `"Go"`, `"Postgres"` $\to$ `"PostgreSQL"`).
3. **Hot-Reloadable Extraction**: Load taxonomy definitions dynamically on `MetadataExtractor` initialization, enabling zero-code updates to recruitment skills.

## Consequences
- **Positive**: Eliminates silent extraction misses; enables recruiter-customizable skill dictionaries; normalizes abbreviations into canonical skill names.
- **Negative**: Requires maintaining and validating the taxonomy YAML schema.
