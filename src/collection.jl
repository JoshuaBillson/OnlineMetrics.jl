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
struct MetricCollection{M}
    metrics::M
end

AbstractTrees.children(x::MetricCollection) = x.metrics
AbstractTrees.nodevalue(::MetricCollection) = MetricCollection

function Base.show(io::IO, x::MetricCollection)
    print_tree(io, x)
end

function step!(x::MetricCollection, ŷ, y)
    foreach(metric -> step!(metric, ŷ, y), x.metrics)
end