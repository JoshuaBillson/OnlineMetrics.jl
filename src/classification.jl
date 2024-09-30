"""
Classification metrics are used to evaluate the performance of models that predict a
discrete label for each observation. Subtypes should implement an `batchstate` method which
assumes that both `ŷ` and `y` are encoded as integers.
"""
abstract type ClassificationMetric <: AbstractMetric end

function update(x::ClassificationMetric, state, ŷ::AbstractArray{<:AbstractFloat,N}, y::AbstractArray{<:Real,N}) where {N}
    @assert size(ŷ) == size(y)
    class_dim = max(N - 1, 1)
    if (N == 1) || (size(ŷ, class_dim) == 1)
        return update(x, state, round.(Int, ŷ), round.(Int, y))
    end
    return update(x, state, _onecold(ŷ, class_dim), _onecold(y, class_dim))
end

# Accuracy

"""
    Accuracy()
    
Measures the model's overall accuracy as `correct / total`.
"""
struct Accuracy <: ClassificationMetric end

name(::Type{Accuracy}) = "accuracy"

init(::Accuracy) = (correct=0, total=0)

function update(::Accuracy, state, ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    return (correct = state.correct + sum(ŷ .== y), total = state.total + length(ŷ))
end

compute(::Accuracy, state) = state.correct / max(state.total, 1)

# MIoU

"""
    MIoU(nclasses::Int)

Mean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label.
This measure is frequently used for segmentation models.
"""
struct MIoU <: ClassificationMetric
    nclasses::Int
end

MIoU(x::AbstractVector) = MIoU(vec(x))

name(::Type{MIoU}) = "MIoU"

init(x::MIoU) = (intersection=zeros(Int, x.nclasses), union=zeros(Int, x.nclasses))

function update(x::MIoU, state, ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    intersection = [sum((ŷ .== cls) .&& (y .== cls)) for cls in _classes(x.nclasses)]
    union = [sum((ŷ .== cls) .|| (y .== cls)) for cls in _classes(x.nclasses)]
    return (intersection = state.intersection + intersection, union = state.union + union)
end

compute(x::MIoU, state) = sum((state.intersection .+ eps(Float64)) ./ (state.union .+ eps(Float64))) / x.nclasses

"""
    ConfusionMatrix(nclasses::Int)

Calculate the confusion matrix over two or more classes. The columns of the resulting `nclasses x nclasses`
matrix correspond to the true label while the rows correspond to the prediction.

# Arguments
- `nclasses::Int`: The number of possible classes in the classification task.
"""
struct ConfusionMatrix <: ClassificationMetric
    nclasses::Int
end

name(::Type{ConfusionMatrix}) = "confusion"

init(x::ConfusionMatrix) = (;confusion=zeros(Int, x.nclasses, x.nclasses))

function update(x::ConfusionMatrix, state, ŷ::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
    return update(x, state, reshape(ŷ, (1,:)), reshape(y, (1,:)))
end
function update(x::ConfusionMatrix, state, ŷ::AbstractArray{<:Integer,4}, y::AbstractArray{<:Integer,4})
    return update(x, state, _flatten(ŷ), _flatten(y))
end
function update(x::ConfusionMatrix, state, ŷ::AbstractArray{<:Integer,2}, y::AbstractArray{<:Integer,2})
    confusion = _onehot(ŷ, _classes(x.nclasses)) * transpose(_onehot(y, _classes(x.nclasses)))
    return (;confusion = state.confusion .+ confusion)
end

compute(::ConfusionMatrix, state) = state.confusion

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
struct Precision <: ClassificationMetric
    agg::Symbol
    nclasses::Int
end

Precision(nclasses::Int; agg=:macro) = Precision(agg, nclasses)

name(::Type{Precision}) = "precision"

init(x::Precision) = (tp=zeros(Int, x.nclasses), fp=zeros(Int, x.nclasses))

function update(x::Precision, state, ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    TP = [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    FP = [_fp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    return (tp = state.tp + TP, fp = state.fp + FP)
end

function compute(x::Precision, state)
    TP, FP = state
    @match x.agg begin
        :macro => mean(TP ./ (TP .+ FP .+ eps(Float64)))
        :micro => mean(TP) / (mean(TP) + mean(FP) + eps(Float64))
        :nothing => TP ./ (TP .+ FP .+ eps(Float64))
    end
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
struct Recall <: ClassificationMetric
    agg::Symbol
    nclasses::Int
end

Recall(nclasses::Int; agg=:macro) = Recall(agg, nclasses)

name(::Type{Recall}) = "recall"

init(x::Recall) = (tp=zeros(Int, x.nclasses), fn=zeros(Int, x.nclasses))

function update(x::Recall, state, ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    TP = [_tp(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    FN = [_fn(ŷ .== i, y .== i) for i in _classes(x.nclasses)]
    return (tp = state.tp + TP, fn = state.fn + FN)
end

function compute(x::Recall, state)
    TP, FN = state
    @match x.agg begin
        :macro => mean(TP ./ (TP .+ FN .+ eps(Float64)))
        :micro => mean(TP) / (mean(TP) + mean(FN) + eps(Float64))
        :nothing => TP ./ (TP .+ FN .+ eps(Float64))
    end
end