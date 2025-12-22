"""
    AverageMeasure(measure::Function, name::String)

Tracks the average value of a given measure over mini-batches.
Typically used to track loss functions or regression metrics.
"""
struct AverageMeasure{M} <: AbstractMetric
    name::String
    measure::M

    AverageMeasure(measure::Function, name::String) = new{typeof(measure)}(name, measure)
end

name(x::AverageMeasure) = x.name

initial_state(::AverageMeasure) = (;n=0, avg=0.0)

function batch_state(x::AverageMeasure, y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real})
    return (;n=length(y_pred), avg=mean(x.measure(y_pred, y_true)))
end

function merge_state(::AverageMeasure, state1, state2)
    n1, avg1 = state1.n, state1.avg
    n2, avg2 = state2.n, state2.avg
    n = n1 + n2
    avg = (avg1 * (n1 / n)) + (avg2 * (n2 / n))
    return (;n, avg)
end

metric_value(::AverageMeasure, state) = state.avg