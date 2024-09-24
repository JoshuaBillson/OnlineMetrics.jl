"""
    MetricCollection(metrics...; prefix="")

An object to track one or more metrics. Each metric is associated with a unique name, 
which defaults to `name(metric)`. This can be overriden by providing a `name => metric`
pair. The `prefix` keyword adds a constant prefix string to every name.

# Example 1
```jldoctest
julia> md = MetricCollection(Accuracy(), MIoU([0,1]); prefix="train_");

julia> step!(md, [0, 0, 1, 0], [0, 0, 1, 1]);

julia> md
MetricCollection(train_accuracy=0.75, train_MIoU=0.5833333333333334)

julia> step!(md, [0, 0, 1, 1], [0, 0, 1, 1]);

julia> md
MetricCollection(train_accuracy=0.875, train_MIoU=0.775)

julia> reset!(md)

julia> md
MetricCollection(train_accuracy=0.0, train_MIoU=1.0)
```

# Example 2
```jldoctest
julia> md = MetricCollection("train_acc" => Accuracy(), "val_acc" => Accuracy());

julia> step!(md, "train_acc", [0, 1, 1, 0], [1, 1, 1, 0]);  # update train_acc

julia> step!(md, r"val_", [1, 1, 1, 0], [1, 1, 1, 0]);  # update metrics matching regex

julia> md
MetricCollection(train_acc=0.75, val_acc=1.0)
```
"""
struct MetricCollection{D<:OrderedDict}
    metrics::D
end

function MetricCollection(metrics...; prefix="")
    # Preprocess Metric Names
    metrics = @pipe map(_preprocess_metric, metrics) |> [(prefix * n) => m for (n, m) in _]

    # Populate Metric Dict
    metric_dict = OrderedDict{String,Any}()
    for (name, metric) in metrics
        metric_dict[name] = Metric(metric)
    end

    # Return MetricCollection
    return MetricCollection(metric_dict)
end

_preprocess_metric(m::Pair{String, <:AbstractMetric}) = m
_preprocess_metric(m::Pair{:Symbol, <:AbstractMetric}) = string(first(m)) => last(m)
_preprocess_metric(m::AbstractMetric) = name(m) => m

Base.keys(x::MetricCollection) = keys(x.metrics)
Base.getindex(x::MetricCollection, i) = x.metrics[i]
Base.setindex!(x::MetricCollection, val, key...) = Base.setindex!(x.metrics, val, keys...)
Base.pairs(x::MetricCollection) = Base.pairs(x.metrics)

function Base.show(io::IO, ::MIME"text/plain", x::MetricCollection)
    printstyled(io, "MetricCollection(")
    for (i, (name, metric)) in enumerate(x.metrics)
        printstyled(io, "$name")
        printstyled(io, "=")
        printstyled(io, "$(compute(metric))")
        i < length(x.metrics) && printstyled(io, ", ")
    end
    printstyled(io, ")")
end

"""
    step!(x::MetricCollection, ŷ, y)
    step!(x::MetricCollection, metric::String, ŷ, y)
    step!(x::MetricCollection, metric::Regex, ŷ, y)

Update the metric for the current epoch using the provided prediction/label pair.
"""
step!(md::MetricCollection, ŷ, y) = step!(md, r".*", ŷ, y)
step!(md::MetricCollection, metric::String, ŷ, y) = step!(md.metrics[metric], ŷ, y)
function step!(md::MetricCollection, pattern::Regex, ŷ, y)
    foreach(metric -> step!(md, metric, ŷ, y), _filter_metrics(md, pattern))
end

_filter_metrics(md::MetricCollection, pat::Regex) = filter(x -> contains(x, pat), keys(md.metrics))

reset!(md::MetricCollection) = foreach(reset!, values(md.metrics))

function scores(md::MetricCollection)
    names = keys(md.metrics) .|> Symbol |> Tuple
    vals = map(compute, values(md.metrics))
    return NamedTuple{names}(vals)
end