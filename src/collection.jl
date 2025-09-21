"""
    MetricCollection(metrics...)

An object to track one or more metrics concurrently.

# Example
```jldoctest
julia> mc = MetricCollection(Accuracy(2), MIoU(2));

julia> step!(mc, [0, 0, 1, 0], [0, 0, 1, 1]);

julia> mc
MetricCollection
├─ :accuracy
│  ├─ :value ⇒ 0.75
│  ├─ :n ⇒ 4
│  └─ :correct ⇒ 3
└─ :MIoU
   ├─ :value ⇒ 0.583333
   ├─ :union ⇒ [3, 2]
   └─ :intersection ⇒ [2, 1]
```
"""
struct MetricCollection{M}
    metrics::M
    MetricCollection(metrics...) = MetricCollection(metrics)
    MetricCollection(metrics::T) where {T <: Tuple} = new{T}(metrics)
end

function step!(x::MetricCollection, ŷ, y)
    foreach(metric -> step!(metric, ŷ, y), x.metrics)
end

function value(m::MetricCollection)
    return NamedTuple([Symbol(name(metric)) => value(metric) for metric in m.metrics])
end

params(m::MetricCollection) = map(params, m.metrics)

name(::MetricCollection) = "MetricCollection"

AbstractTrees.children(x::MetricCollection) = x.metrics
AbstractTrees.nodevalue(::MetricCollection) = MetricCollection

function Base.show(io::IO, x::MetricCollection)
    print_tree(io, x)
end