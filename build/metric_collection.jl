"""
    merge(metrics...)

Merge muliple `Metric` or `MetricCollection` objects into a single object.

Useful for aggregating metrics across multiple devices or processes.
"""
function merge end

"""
    step!(x::Metric, y_pred, y_true)
    step!(x::MetricCollection, y_pred, y_true)

Update the metric(s) with a new batch of predictions and true labels.
"""
function step! end

"""
    value(x::Metric)
    value(x::MetricCollection)

Compute the current value of the metric(s).
"""
function value end

"""
    Metric(m::AbstractMetric; name=name(m))

A struct to track a single metric and its state over multiple mini-batches.

# Parameters
- `m::AbstractMetric`: A metric to be tracked.
- `name::String`: An optional name for the metric. Defaults to the name of the metric type.

# Example
```julia
julia> m = Metric(Accuracy(2); name="My Accuracy Metric")
julia> step!(m, [0,1,0,1], [0,1,1,1])
julia> value(m)
0.75
```
"""
mutable struct Metric{M<:AbstractMetric,S}
    name::String
    metric::M
    state::S

    Metric(name, metric, state) = new{typeof(metric), typeof(state)}(name, metric, state)
    Metric(m::AbstractMetric; name=name(m)) = Metric(name, m, initial_state(m))
end

function step!(x::Metric, y_pred, y_true)
    x.state = step(x.metric, y_pred, y_true, x.state)
    return x
end

value(x::Metric) = current_value(x.metric, x.state)

function merge(xs::Vararg{M}) where M<:Metric
    @argcheck length(xs) > 0 "At least one Metric must be provided to merge."
    newstate = merge_state(xs[1].metric, [x.state for x in xs]...)
    return Metric(xs[1].name, xs[1].metric, newstate)
end

function Base.show(io::IO, x::Metric)
    print(io, x.name, ": ")
    print(io, "value=", value(x), " | ")
    print(io, join(["$k=$v" for (k,v) in pairs(x.state)], " | "))
end

"""
    MetricCollection(metrics...)

An object to track one or more metrics concurrently.

# Parameters
- `metrics...`: A variable number of metrics to be tracked. Each metric can be provided
  either as an instance of a subtype of `AbstractMetric` or as a `Pair{String, AbstractMetric}`
  to override the default name of the metric.

# Example
```julia
julia> m = MetricCollection(Accuracy(2), "precision" => Precision(2, agg=nothing), mIoU(2))
MetricCollection
├─ accuracy: value=0.0 | correct=0 | n=0
├─ precision: value=[1.0, 1.0] | tp=[0, 0] | fp=[0, 0]
└─ mIoU: value=1.0 | intersection=[0, 0] | union=[0, 0]


julia> step!(m, [0, 0, 1, 0], [0, 0, 1, 1])
MetricCollection
├─ accuracy: value=0.75 | correct=3 | n=4
├─ precision: value=[0.666667, 1.0] | tp=[2, 1] | fp=[1, 0]
└─ mIoU: value=0.583333 | intersection=[2, 1] | union=[3, 2]
```
"""
struct MetricCollection{M}
    metrics::M

    MetricCollection(metrics::Vararg{M}) where {M<:Metric} = new{typeof(metrics)}(metrics)
    function MetricCollection(metrics...)
        metrics = map(_build_metric, metrics)
        return new{typeof(metrics)}(metrics)
    end
end

_build_metric(m::AbstractMetric) = Metric(m)
_build_metric(m::Pair{String,<:AbstractMetric}) = Metric(m[2]; name=m[1])

function step!(x::MetricCollection, ŷ, y)
    foreach(metric -> step!(metric, ŷ, y), x.metrics)
    return x
end

function value(m::MetricCollection)
    return NamedTuple([Symbol(metric.name) => value(metric) for metric in m.metrics])
end

function merge(xs::Vararg{MC}) where MC<:MetricCollection
    @argcheck length(xs) > 0 "At least one MetricCollection must be provided to merge."
    merged_metrics = Metric[]
    for i in eachindex(xs[1].metrics)
        metrics_to_merge = [x.metrics[i] for x in xs]
        push!(merged_metrics, merge(metrics_to_merge...))
    end
    return MetricCollection(merged_metrics...)
end

AbstractTrees.children(x::MetricCollection) = x.metrics
AbstractTrees.nodevalue(::MetricCollection) = MetricCollection

function Base.show(io::IO, x::MetricCollection)
    print_tree(io, x)
end