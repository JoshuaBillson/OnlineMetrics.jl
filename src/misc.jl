"""
    AverageMeasure(measure::Function)

Tracks the average of the given measure over mini-batches.
"""
mutable struct AverageMeasure{M} <: AbstractMetric
    lock::ReentrantLock
    measure::M
    n::Int
    avg::Float64
end

AverageMeasure(measure::Function) = AverageMeasure(ReentrantLock(), measure, 0, 0.0)

function step!(x::AverageMeasure, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    lock(x.lock) do
        x.n += 1
        x.avg = (x.avg * (x.n-1) / x.n) + (x.measure(ŷ, y) / x.n)
    end
end

value(x::AverageMeasure) = x.avg

params(x::AverageMeasure) = (;x.n)