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
    
Measures the model's overall accuracy as `correct / total`.
"""
mutable struct Accuracy <: ClassificationMetric
    correct::Int
    total::Int
    Accuracy() = new(0,0)
end

name(::Accuracy) = "accuracy"

function step!(m::Accuracy, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    m.correct += sum(ŷ .== y)
    m.total += length(ŷ)
end

value(m::Accuracy) = m.correct / max(m.total, 1)

params(x::Accuracy) = (;n=x.total, x.correct)

# MIoU

"""
    MIoU(nclasses::Int)

Mean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label.
This measure is frequently used for segmentation models.
"""
mutable struct MIoU <: ClassificationMetric
    nclasses::Int
    intersection::Vector{Int}
    union::Vector{Int}
end

function MIoU(nclasses::Int)
    @argcheck nclasses >= 1
    return MIoU(nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

name(::MIoU) = "MIoU"

function step!(x::MIoU, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    x.intersection += [sum((ŷ .== cls) .& (y .== cls)) for cls in 0:x.nclasses-1]
    x.union += [sum((ŷ .== cls) .| (y .== cls)) for cls in 0:x.nclasses-1]
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
    nclasses::Int
    confusion::Matrix{Int}
end

function ConfusionMatrix(nclasses::Int)
    @argcheck nclasses > 0
    return ConfusionMatrix(nclasses, zeros(Int, nclasses, nclasses))
end

name(::ConfusionMatrix) = "confusion"

function params(x::ConfusionMatrix)
    tp, tn, fp, fn = _tfpn(x.confusion)
    return (;tp, tn, fp, fn)
end

function step!(x::ConfusionMatrix, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    return step!(x, reshape(ŷ, (1,:)), reshape(y, (1,:)))
end
function step!(x::ConfusionMatrix, ŷ::AbstractArray{<:Integer,4}, y::AbstractArray{<:Integer,4})
    return step!(x,  _flatten(ŷ), _flatten(y))
end
function step!(x::ConfusionMatrix, ŷ::AbstractArray{<:Integer,2}, y::AbstractArray{<:Integer,2})
    classes = collect(0:x.nclasses-1)
    x.confusion .+= _onehot(ŷ, classes) * transpose(_onehot(y, classes))
end

value(x::ConfusionMatrix) = 0#x.confusion

"""
    Precision(nclasses::Int; agg=:macro)

Precision is the ratio of true positives to the sum of true positives and false positives, measuring the accuracy of positive predictions.

# Arguments
- `classes::Vector{Int}`: A vector of integer class labels representing the possible classes for the classification task.

# Keyword Arguments
- `agg`: Specifies the type of precision aggregation to be computed. The possible values are:
  - `:macro`: Calculates macro-averaged precision, which computes the precision for each class independently and then takes the average.
  - `:micro`: Calculates micro-averaged precision, which aggregates the contributions of all classes to compute a single precision value.
  - `:nothing`: Calculates the per-class precision, which is returned as a `Vector` with the same length as `classes`.
"""
mutable struct Precision <: ClassificationMetric
    agg::Symbol
    nclasses::Int
    tp::Vector{Int}
    fp::Vector{Int}
end

function Precision(nclasses::Int; agg=:macro)
    @argcheck nclasses > 0
    @argcheck agg in (:macro, :micro, nothing)
    Precision(Symbol(agg), nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

name(::Precision) = "precision"

params(x::Precision) = (;x.agg, tp=x.tp, fp=x.fp)

function step!(x::Precision, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    x.tp .+= [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    x.fp .+= [_fp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
end

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
    Recall(nclasses::Int; agg=:macro)

Recall, also known as sensitivity or true positive rate, is the ratio of true positives to the sum of true positives and false negatives, measuring the ability of the classifier to identify all positive instances.

# Arguments
- `classes::Vector{Int}`: A vector of integer class labels representing the possible classes for the classification task.

# Keyword Arguments
- `agg`: Specifies the type of recall aggregation to be computed. The possible values are:
  - `:macro`: Calculates macro-averaged recall, which computes the recall for each class independently and then takes the average.
  - `:micro`: Calculates micro-averaged recall, which aggregates the contributions of all classes to compute a single recall value.
  - `:nothing`: Calculates the per-class recall, which is returned as a `Vector` with the same length as `classes`.
"""
mutable struct Recall <: ClassificationMetric
    agg::Symbol
    nclasses::Int
    tp::Vector{Int}
    fn::Vector{Int}
end

function Recall(nclasses::Int; agg=:macro)
    @argcheck nclasses > 0
    @argcheck agg in (:macro, :micro, nothing)
    Recall(Symbol(agg), nclasses, zeros(Int, nclasses), zeros(Int, nclasses))
end

name(::Recall) = "recall"

function step!(x::Recall, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    x.tp .+= [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    x.fn .+= [_fn(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
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