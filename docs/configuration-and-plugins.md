# Configuration and plugins

`recsys.conf` is packaged in the wheel and composed with
`initialize_config_module(config_module="recsys.conf")`. Structured dataclasses validate
all stable platform sections while plugin `parameters` dictionaries remain extensible.

## Precedence

Configuration is resolved in this order:

1. packaged defaults;
2. an optional external YAML file passed with `--config`;
3. CLI dotlist overrides.

Examples:

```bash
recsys --config examples/configs/tabular.yaml data prepare
recsys run model=lightgcn training.device=cpu training.epochs=3
recsys -o evaluation.top_k=20 -o model.parameters.layers=3 train
```

`--workspace-root` is global and defaults to the current directory. Relative dataset paths
resolve against it; generated state always resolves below its `var/` directory.

## Registries

The CLI exposes the active built-ins:

```bash
recsys plugins list
```

The stable extension contracts are:

- model and retriever: accept `TrainingContext`, return `RuntimeSpec`;
- fusion: accept named score vectors and return one fused score vector;
- reranker: accept history and candidates, return reordered candidates;
- runtime: load verified arrays/files and score a `RuntimeQuery`.

Plugin code owns its parameters and payload meaning. The artifact layer owns identity,
checksums, catalog mappings, and portability. A plugin must declare `known_user`, `history`,
or both. Hybrid capability is the intersection of its configured components.

The Ollama adapter is optional. Its endpoint, model, prompt template, and timeout are all
configuration values; no language model is required by the default pipeline.
