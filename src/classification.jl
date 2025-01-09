"""
Classification metrics are used to evaluate the performance of models that predict a
discrete label for each observation. Subtypes should implement an `batchstate` method which
assumes that both `ŷ` and `y` are encoded as integers.
"""
abstract type ClassificationMetric <: AbstractMetric end

function step!(x::ClassificationMetric, ŷ::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return step!(x, _onecold(ŷ), _onecold(y))
end

# Accuracy

"""
    Accuracy()
    
Measures the model's overall accuracy as `correct / n`.
"""
mutable struct Accuracy <: ClassificationMetric
    lock::ReentrantLock
    correct::Int
    n::Int
    Accuracy() = new(ReentrantLock(), 0,0)
end

function step!(m::Accuracy, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    lock(m.lock) do
        m.correct += sum(ŷ .== y)
        m.n += length(ŷ)
    end
end

value(m::Accuracy) = m.correct / max(m.n, 1)

params(x::Accuracy) = (;x.n, x.correct)

# MIoU

"""
    MIoU(nclasses::Int)

Mean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label.
This measure is frequently used for segmentation models.
"""
mutable struct MIoU <: ClassificationMetric
    lock::ReentrantLock
    nclasses::Int
    intersection::Vector{Int}
    union::Vector{Int}
end

function MIoU(nclasses::Int)
    @argcheck nclasses >= 1
    return MIoU(ReentrantLock(), nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

function step!(x::MIoU, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    lock(x.lock) do
        x.intersection += [sum((ŷ .== cls) .& (y .== cls)) for cls in 0:x.nclasses-1]
        x.union += [sum((ŷ .== cls) .| (y .== cls)) for cls in 0:x.nclasses-1]
    end
end

value(x::MIoU) = round(sum((x.intersection .+ eps(Float64)) ./ (x.union .+ eps(Float64))) / x.nclasses, digits=15)

params(x::MIoU) = (;union=x.union, intersection=x.intersection)

"""
    ConfusionMatrix(nclasses::Int)

Calculate the confusion matrix over two or more classes. The columns of the resulting `nclasses x nclasses`
matrix correspond to the true label while the rows correspond to the prediction.

# Arguments
- `nclasses::Int`: The number of possible classes in the classification task.
"""
mutable struct ConfusionMatrix <: ClassificationMetric
    lock::ReentrantLock
    nclasses::Int
    confusion::Matrix{Int}
end

function ConfusionMatrix(nclasses::Int)
    @argcheck nclasses > 0
    return ConfusionMatrix(ReentrantLock(), nclasses, zeros(Int, nclasses, nclasses))
end

function step!(x::ConfusionMatrix, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    lock(x.lock) do
        classes = collect(0:x.nclasses-1)
        x.confusion .+= _onehot(ŷ, classes) * transpose(_onehot(y, classes))
    end
end

value(x::ConfusionMatrix) = x.confusion

function params(x::ConfusionMatrix)
    tp, tn, fp, fn = _tfpn(x.confusion)
    return (;tp, tn, fp, fn)
end

"""
    Precision(nclasses::Int; agg=:macro)

Precision is the ratio of true positives to the sum of true positives and false positives, measuring the accuracy of positive predictions.

# Arguments
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Arguments
- `agg`: Specifies the type of precision aggregation to be computed. The possible values are:
- `:macro`: Calculates macro-averaged precision, which computes the precision for each class independently and then takes the average.
- `:micro`: Calculates micro-averaged precision, which aggregates the contributions of all classes to compute a single precision value.
- `:nothing`: Calculates the per-class precision, which is returned as a `Vector` with the same length as `classes`.
"""
mutable struct Precision <: ClassificationMetric
    lock::ReentrantLock
    agg::Symbol
    nclasses::Int
    tp::Vector{Int}
    fp::Vector{Int}
end

function Precision(nclasses::Int; agg=:macro)
    @argcheck nclasses > 0
    @argcheck agg in (:macro, :micro, nothing)
    Precision(ReentrantLock(), Symbol(agg), nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

function step!(x::Precision, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    lock(x.lock) do
        x.tp .+= [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
        x.fp .+= [_fp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    end
end

params(x::Precision) = (;x.agg, tp=x.tp, fp=x.fp)

function value(x::Precision)
    ϵ = eps(Float64)
    v = @match x.agg begin
        :macro => mean((x.tp .+ ϵ) ./ (x.tp .+ x.fp .+ ϵ))
        :micro => mean(x.tp .+ ϵ) / (mean(x.tp) + mean(x.fp) + ϵ)
        :nothing => (x.tp .+ ϵ) ./ (x.tp .+ x.fp .+ ϵ)
    end
    return round.(v, digits=15)
end

"""
    BinaryPrecision()

A variant of `Precision` specialized for binary classification.
"""
mutable struct BinaryPrecision <: ClassificationMetric
    precision::Precision
    BinaryPrecision() = new(Precision(2; agg=nothing))
end

function step!(x::BinaryPrecision, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    step!(x.precision, ŷ, y)
end

params(x::BinaryPrecision) = (;tp=x.precision.tp[2], fp=x.precision.fp[2])

value(x::BinaryPrecision) = value(x.precision)[2]

"""
    Recall(nclasses::Int; agg=:macro)

Recall, also known as sensitivity or true positive rate, is the ratio of true positives to the sum of true positives and false negatives, measuring the ability of the classifier to identify all positive instances.

# Arguments
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Arguments
- `agg`: Specifies the type of recall aggregation to be computed. The possible values are:
- `:macro`: Calculates macro-averaged recall, which computes the recall for each class independently and then takes the average.
- `:micro`: Calculates micro-averaged recall, which aggregates the contributions of all classes to compute a single recall value.
- `:nothing`: Calculates the per-class recall, which is returned as a `Vector` with the same length as `classes`.
"""
mutable struct Recall <: ClassificationMetric
    lock::ReentrantLock
    agg::Symbol
    nclasses::Int
    tp::Vector{Int}
    fn::Vector{Int}
end

function Recall(nclasses::Int; agg=:macro)
    @argcheck nclasses > 0
    @argcheck agg in (:macro, :micro, nothing)
    Recall(ReentrantLock(), Symbol(agg), nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

function step!(x::Recall, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    lock(x.lock) do
        x.tp .+= [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
        x.fn .+= [_fn(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    end
end

function value(x::Recall)
    ϵ = eps(Float64)
    v = @match x.agg begin
        :macro => mean((x.tp .+ ϵ) ./ (x.tp .+ x.fn .+ ϵ))
        :micro => mean(x.tp .+ ϵ) / (mean(x.tp) + mean(x.fn) + ϵ)
        :nothing => (x.tp .+ ϵ) ./ (x.tp .+ x.fn .+ ϵ)
    end
    return round.(v, digits=15)
end

params(x::Recall) = (;x.agg, tp=x.tp, fn=x.fn)

"""
    BinaryRecall()

A variant of `Recall` specialized for binary classification.
"""
mutable struct BinaryRecall <: ClassificationMetric
    recall::Recall
    BinaryRecall() = new(Recall(2; agg=nothing))
end

function step!(x::BinaryRecall, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    step!(x.recall, ŷ, y)
end

params(x::BinaryRecall) = (;tp=x.recall.tp[2], fn=x.recall.fn[2])

value(x::BinaryRecall) = value(x.recall)[2]