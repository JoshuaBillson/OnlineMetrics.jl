# OnlineMetrics

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/OnlineMetrics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/OnlineMetrics.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/OnlineMetrics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/OnlineMetrics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/OnlineMetrics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/OnlineMetrics.jl)


[OnlineMetrics](https://github.com/JoshuaBillson/OnlineMetrics.jl) provides metrics for online, streaming, and batched machine learning workflows. Metrics implement a small, consistent interface so they can be updated batch-by-batch, merged across devices/processes, and queried for current values with minimal overhead.

## Features
- Compute metrics in batches - no need to store full datasets.
- Track both indivudual and multiple metrics.
- Mergeable states for distributed/parallel aggregation.
- Pluggable data formats (e.g., `OneHot`) with validation and formatting hooks.
- Small, explicit API surface for easy extension.

## Design philosophy
- **Minimal, explicit interface:** Implement `AbstractMetric` and the handful of required functions (`name`, `initial_state`, `batch_state`, `merge_state`, `current_value`) to add a new metric.
- **Robust input handling:** Metrics declare or accept data formats; validation and formatting are separated from core metric logic.
- **Usability:** Friendly `Base.show` representations and `MetricCollection` tree-printing for quick inspection.

## Quickstart

#### Install:
```julia
using Pkg
Pkg.add("OnlineMetrics")
```

#### Basic usage:
```julia
using OnlineMetrics

m = Metric(Accuracy(2))                           # track a single metric
mc = MetricCollection(Accuracy(2), Precision(2))  # track multiple metrics

step!(m, [0,1,1,0], [0,1,0,0])                    # update with a batch
step!(mc, [0,1,1,0], [0,1,0,0])                   # update collection

value(m)                                          # current value of single metric
value(mc)                                         # NamedTuple of values for collection
```

#### Merging states (useful for distributed runs):
```julia
m1 = Metric(Accuracy(2)); step!(m1, preds1, labels1)
m2 = Metric(Accuracy(2)); step!(m2, preds2, labels2)
m_merged = merge(m1, m2)
value(m_merged)
```

## Examples
See the `src` folder for metric implementations and refer to the docs for usage examples.

## Development & testing
Run tests from the package root:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
Or in the package REPL:
```julia
] test OnlineMetrics
```

## Contributing
Contributions welcome. Please open issues or pull requests for bug fixes, new metrics, or API improvements. Follow existing style in src and add tests to runtests.jl.

## License
Distributed under the terms of the MIT license. See the LICENSE file.