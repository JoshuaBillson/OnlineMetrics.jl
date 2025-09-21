"""
Classification metrics are used to evaluate the performance of models that predict a
discrete label for each observation. Subtypes should implement an `batchstate` method which
assumes that both `ŷ` and `y` are encoded as integers.
"""
abstract type ClassificationMetric{N} <: AbstractMetric end

function step!(x::ClassificationMetric{N}, y_pred::AbstractArray{<:Real}, y_true::AbstractArray{<:Real}) where N
    return step!(x, one_hot(y_pred, N), one_hot(y_true, N))
end

# Accuracy

"""
    Accuracy()
    
Measures the model's overall accuracy as `correct / n`.
"""
mutable struct Accuracy{N} <: ClassificationMetric{N}
    correct::Int
    n::Int
    Accuracy(nclasses::Int) = new{nclasses}(0,0)
end

function step!(m::Accuracy, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    m.correct += sum(onecold(y_pred) .== onecold(y_true))
    m.n += size(y_true,2)
end

value(m::Accuracy) = m.correct / max(m.n, 1)

params(x::Accuracy) = (;x.n, x.correct)

name(::Accuracy) = "accuracy"

# MIoU

"""
    MIoU(nclasses::Int)

Mean Intersection over Union (MIoU) is a measure of the overlap between a prediction and a label.
This measure is frequently used for segmentation models.
"""
mutable struct MIoU{N} <: ClassificationMetric{N}
    intersection::Vector{Int}
    union::Vector{Int}
    MIoU(nclasses::Int) = new{nclasses}(zeros(Int, nclasses), zeros(Int, nclasses))
end

function step!(x::MIoU{N}, y_pred::OneHotMatrix, y_true::OneHotMatrix) where N
    ŷ = onecold(y_pred)
    y = onecold(y_true)
    x.intersection += [sum((ŷ .== cls) .& (y .== cls)) for cls in 1:N]
    x.union += [sum((ŷ .== cls) .| (y .== cls)) for cls in 1:N]
end

value(x::MIoU{N}) where N = sum((x.intersection .+ eps(Float64)) ./ (x.union .+ eps(Float64))) / N

params(x::MIoU) = (;union=x.union, intersection=x.intersection)

name(::MIoU) = "MIoU"

"""
    ConfusionMatrix(nclasses::Int)

Calculate the confusion matrix over two or more classes. The columns of the resulting `nclasses x nclasses`
matrix correspond to the true label while the rows correspond to the prediction.

# Arguments
- `nclasses::Int`: The number of possible classes in the classification task.
"""
mutable struct ConfusionMatrix{N} <: ClassificationMetric{N}
    confusion::Matrix{Int}
    ConfusionMatrix(nclasses::Int) = new{nclasses}(zeros(Int, nclasses, nclasses))
end


function step!(x::ConfusionMatrix, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    x.confusion .+= _confusion_matrix(y_pred, y_true)
end

value(x::ConfusionMatrix) = x.confusion

function params(x::ConfusionMatrix)
    tp, tn, fp, fn = _tfpn(x.confusion)
    return (;tp, tn, fp, fn)
end

name(::ConfusionMatrix) = "confusion_matrix"

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
mutable struct Precision{N} <: ClassificationMetric{N}
    agg::Symbol
    tp::Vector{Int}
    fp::Vector{Int}
end

function Precision(nclasses::Int; agg=:macro)
    @argcheck nclasses > 0
    @argcheck agg in (:macro, :micro, nothing)
    Precision{nclasses}(Symbol(agg), zeros(Int, nclasses), zeros(Int, nclasses))
end

function step!(x::Precision, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    TP, _, FP, _ = _tfpn(y_pred, y_true)
    x.tp .+= TP
    x.fp .+= FP
end

params(x::Precision) = (;x.agg, tp=x.tp, fp=x.fp)

function value(x::Precision)
    ϵ = eps(Float64)
    return @match x.agg begin
        :macro => mean((x.tp .+ ϵ) ./ (x.tp .+ x.fp .+ ϵ))
        :micro => mean(x.tp .+ ϵ) / (mean(x.tp) + mean(x.fp) + ϵ)
        :nothing => (x.tp .+ ϵ) ./ (x.tp .+ x.fp .+ ϵ)
    end
end

name(::Precision) = "precision"

"""
    BinaryPrecision()

A variant of `Precision` specialized for binary classification.
"""
mutable struct BinaryPrecision <: ClassificationMetric{2}
    tp::Int
    fp::Int
    BinaryPrecision() = new(0, 0)
end

function step!(x::BinaryPrecision, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    TP, _, FP, _ = _tfpn(y_pred, y_true)
    x.tp += TP[2]
    x.fp += FP[2]
end

params(x::BinaryPrecision) = (;tp=x.tp, fp=x.fp)

value(x::BinaryPrecision) = (x.tp + eps(Float64)) / (x.tp + x.fp .+ eps(Float64))

name(::BinaryPrecision) = "binary_precision"

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
mutable struct Recall{N} <: ClassificationMetric{N}
    agg::Symbol
    tp::Vector{Int}
    fn::Vector{Int}
    function Recall(nclasses::Int; agg=:macro)
        @argcheck nclasses > 0
        @argcheck agg in (:macro, :micro, nothing)
        new{nclasses}(Symbol(agg), zeros(Int, nclasses), zeros(Int, nclasses))
    end
end

function step!(x::Recall, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    TP, _, _, FN = _tfpn(y_pred, y_true)
    x.tp .+= TP
    x.fn .+= FN
end

function value(x::Recall)
    ϵ = eps(Float64)
    return @match x.agg begin
        :macro => mean((x.tp .+ ϵ) ./ (x.tp .+ x.fn .+ ϵ))
        :micro => mean(x.tp .+ ϵ) / (mean(x.tp) + mean(x.fn) + ϵ)
        :nothing => (x.tp .+ ϵ) ./ (x.tp .+ x.fn .+ ϵ)
    end
end

params(x::Recall) = (;x.agg, tp=x.tp, fn=x.fn)

name(::Recall) = "recall"

"""
    BinaryRecall()

A variant of `Recall` specialized for binary classification.
"""
mutable struct BinaryRecall <: ClassificationMetric{2}
    tp::Int
    fn::Int
    BinaryRecall() = new(0,0)
end

function step!(x::BinaryRecall, y_pred::OneHotMatrix, y_true::OneHotMatrix)
    TP, _, _, FN = _tfpn(y_pred, y_true)
    x.tp += TP[2]
    x.fn += FN[2]
end

params(x::BinaryRecall) = (;tp=x.tp, fn=x.fn)

value(x::BinaryRecall) = (x.tp + eps(Float64)) / (x.tp + x.fn + eps(Float64))

name(::BinaryRecall) = "binary_recall"