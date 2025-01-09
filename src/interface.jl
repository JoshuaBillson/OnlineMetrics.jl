"""
Metrics are measures of a model's performance, such as loss, accuracy, or squared error.

Each metric must implement the following interface:
- `name(::Type{metric})`: Returns the human readable name of the metric.
- `init(metric)`: Returns the initial state of the metric as a `NamedTuple`.
- `update(metric, state, ŷ, y)`: Returns the new state given the previous state and a batch.
- `compute(metric, state)`: Computes the metric's value for the current state.

# Example Implementation
```julia
struct Accuracy <: ClassificationMetric end

name(::Type{Accuracy}) = "accuracy"

init(::Accuracy) = (correct=0, total=0)

function update(::Accuracy, state, ŷ, y)
    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

compute(::Accuracy, state) = state.correct / max(state.total, 1)
```
"""
abstract type AbstractMetric end

"""
    name(m::AbstractMetric)

Human readable name of the given performance measure.
"""
name(::AbstractMetric)

"""
    step!(m::AbstractMetric, ŷ, y)

Update the metric state for the given batch of labels and predictions.
"""
function update end

"""
    value(m::AbstractMetric)

Compute the performance measure from the current metric state.
"""
function value end

params(x::AbstractMetric) = (;)

function Base.show(io::IO, x::AbstractMetric)
    print_tree(io, x)
end

struct NodeValue{V}
    val::V
end

AbstractTrees.nodevalue(x::NodeValue) = x.val

AbstractTrees.nodevalue(::T) where {T <: AbstractMetric} = T
AbstractTrees.children(x::AbstractMetric) = map(NodeValue, (;value = value(x), params(x)...))