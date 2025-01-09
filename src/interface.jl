"""
Metrics are measures of a model's performance, such as loss, accuracy, or squared error.

Each metric must implement the following interface:
- `step!(metric, ŷ, y)`: Updates the metric state for the given batch of predictions and labels.
- `value(metric)`: Return the current value of the metric.
- `params(metric)`: Return a `NamedTuple` containing any internal parameters you wish to report.

# Example Implementation
```julia
mutable struct Accuracy <: ClassificationMetric
    correct::Int
    n::Int
    Accuracy() = new(0,0)
end

function step!(m::Accuracy, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    m.correct += sum(ŷ .== y)
    m.n += length(ŷ)
end

value(m::Accuracy) = m.correct / max(m.n, 1)

params(x::Accuracy) = (;x.n, x.correct)
```
"""
abstract type AbstractMetric end

"""
    step!(m::AbstractMetric, ŷ, y)

Update the metric state for the given batch of labels and predictions.
"""
function step! end

"""
    value(m::AbstractMetric)

Compute the performance measure from the current metric state.
"""
function value end

"""
    params(m::AbstractMetric)

Returns any internal parameters used to track the metric's current value.
"""
params(::AbstractMetric) = (;)

function Base.show(io::IO, x::AbstractMetric)
    print_tree(io, x)
end

struct NodeValue{V}
    val::V
end

AbstractTrees.nodevalue(x::NodeValue) = x.val

AbstractTrees.nodevalue(::T) where {T <: AbstractMetric} = T
AbstractTrees.children(x::AbstractMetric) = map(NodeValue, (;value = value(x), params(x)...))