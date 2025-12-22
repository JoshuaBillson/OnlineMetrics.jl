"""
Classification metrics are used to evaluate the performance of models that predict a
discrete label for each observation. Subtypes of `ClassificationMetric` have a default
data format of `OneHot`, which provides `batch_state` with `Matrix{Bool}` inputs
representing one-hot encoded class labels.
"""
abstract type ClassificationMetric{N} <: AbstractMetric end

data_format(::ClassificationMetric{N}) where N = OneHot(N)

"""
    Accuracy(nclasses::Int)
    
Measures the model's overall accuracy as `correct / n`.

# Parameters
- `nclasses::Int`: The number of classes for the classification task.
"""
struct Accuracy{N} <: ClassificationMetric{N}
    Accuracy(nclasses::Int) = new{nclasses}()
end

name(::Accuracy) = "accuracy"

initial_state(::Accuracy) = (;correct=0, n=0)

function batch_state(::Accuracy{N}, y_pred, y_true) where N
    n = size(y_true, 2)
    correct = sum(argmax(y_pred, dims=1) .== argmax(y_true, dims=1))
    return (;correct, n)
end

merge_state(::Accuracy{N}, state1, state2) where N = (;correct=state1.correct + state2.correct, n=state1.n + state2.n)

metric_value(::Accuracy, state) = state.correct / max(state.n, 1)

"""
    IoU(nclasses::Int; agg=nothing)

Intersection over Union (IoU) is a measure of the overlap between a prediction and a label.

# Parameters
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Parameters
- `agg`: Specifies the type of IoU aggregation to be computed. The possible values are:
    - `:mean`: Calculates mean IoU (mIoU) by averaging the IoU across all classes.
    - `nothing`: Calculates the per-class IoU, which is returned as a `Vector` with the same length as `classes`.
"""
struct IoU{N} <: ClassificationMetric{N}
    agg::Symbol

    function IoU(nclasses::Int; agg=nothing)
        @argcheck agg in (nothing, :mean)
        new{nclasses}(Symbol(agg))
    end
end

name(x::IoU) = x.agg == :mean ? "mIoU" : "IoU"

initial_state(::IoU{N}) where N = (;intersection=zeros(Int, N), union=zeros(Int, N))

function batch_state(::IoU{N}, y_pred, y_true) where N
    intersection = @pipe (y_pred .&& y_true) |> sum(_, dims=2) |> vec
    union = @pipe (y_pred .|| y_true) |> sum(_, dims=2) |> vec
    return (;intersection, union)
end

function merge_state(::IoU, state1, state2)
    return (;intersection=state1.intersection .+ state2.intersection, union=state1.union .+ state2.union)
end

function metric_value(x::IoU{N}, state) where N
    @match x.agg begin
        :mean => sum((state.intersection .+ eps(Float64)) ./ (state.union .+ eps(Float64))) / N
        :nothing => (state.intersection .+ eps(Float64)) ./ (state.union .+ eps(Float64))
    end
end

"""
    mIoU(nclasses::Int)

A convenience constructor for `IoU` that defaults to mean aggregation.
"""
mIoU(nclasses::Int) = IoU(nclasses; agg=:mean)

"""
    ConfusionMatrix(nclasses::Int)

Calculate the confusion matrix over two or more classes. The columns of the resulting `nclasses x nclasses`
matrix correspond to the true label while the rows correspond to the prediction.

# Parameters
- `nclasses::Int`: The number of possible classes in the classification task.
"""
struct ConfusionMatrix{N} <: ClassificationMetric{N}
    ConfusionMatrix(nclasses::Int) = new{nclasses}()
end

name(::ConfusionMatrix) = "confusion_matrix"

initial_state(::ConfusionMatrix{N}) where N = (;confusion=zeros(Int, N, N))

batch_state(::ConfusionMatrix, y_pred, y_true) = (;confusion=_confusion_matrix(y_pred, y_true))

merge_state(::ConfusionMatrix, state1, state2) = (;confusion=state1.confusion .+ state2.confusion)

metric_value(::ConfusionMatrix, state) = state.confusion

"""
    Precision(nclasses::Int; agg=:macro)

Precision is the ratio of true positives to the sum of true positives and false positives, measuring the accuracy of positive predictions.

# Parameters
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Parameters
- `agg`: Specifies the type of precision aggregation to be computed. The possible values are:
    - `:macro`: Calculates macro-averaged precision, which computes the precision for each class independently and then takes the average.
    - `:micro`: Calculates micro-averaged precision, which aggregates the contributions of all classes to compute a single precision value.
    - `nothing`: Calculates the per-class precision, which is returned as a `Vector` with the same length as `classes`.
"""
struct Precision{N} <: ClassificationMetric{N}
    agg::Symbol

    function Precision(nclasses::Int; agg=:macro)
        @argcheck nclasses > 0
        @argcheck agg in (:macro, :micro, nothing)
        new{nclasses}(Symbol(agg))
    end
end

function name(m::Precision)
    @match m.agg begin
        :macro => "macro_precision"
        :micro => "micro_precision"
        :nothing => "per_class_precision"
    end
end

initial_state(::Precision{N}) where N = (;tp=zeros(Int, N), fp=zeros(Int, N))

function batch_state(::Precision{N}, y_pred, y_true) where N
    tp, _, fp, _ = _tfpn(y_pred, y_true)
    return (;tp, fp)
end

merge_state(::Precision, state1, state2) = (;tp=state1.tp .+ state2.tp, fp=state1.fp .+ state2.fp)

metric_value(x::Precision, state) = _precision(state.tp, state.fp; agg=x.agg)

"""
    BinaryPrecision()

A variant of `Precision` specialized for binary classification.
"""
struct BinaryPrecision <: ClassificationMetric{2} end

name(::BinaryPrecision) = "binary_precision"

initial_state(::BinaryPrecision) = (;tp=0, fp=0)

function batch_state(::BinaryPrecision, y_pred, y_true)
    TP, _, FP, _ = _tfpn(y_pred, y_true)
    return (;tp=TP[2], fp=FP[2])
end

merge_state(::BinaryPrecision, state1, state2) = (;tp=state1.tp + state2.tp, fp=state1.fp + state2.fp)

metric_value(::BinaryPrecision, state) = (state.tp + eps(Float64)) / (state.tp + state.fp + eps(Float64))

"""
    Recall(nclasses::Int; agg=:macro)

Recall, also known as sensitivity or true positive rate, is the ratio of true positives to the sum of true positives and false negatives, measuring the ability of the classifier to identify all positive instances.

# Parameters
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Parameters
- `agg`: Specifies the type of recall aggregation to be computed. The possible values are:
    - `:macro`: Calculates macro-averaged recall, which computes the recall for each class independently and then takes the average.
    - `:micro`: Calculates micro-averaged recall, which aggregates the contributions of all classes to compute a single recall value.
    - `:nothing`: Calculates the per-class recall, which is returned as a `Vector` with the same length as `classes`.
"""
struct Recall{N} <: ClassificationMetric{N}
    agg::Symbol

    function Recall(nclasses::Int; agg=:macro)
        @argcheck nclasses > 0
        @argcheck agg in (:macro, :micro, nothing)
        new{nclasses}(Symbol(agg))
    end
end

function name(m::Recall)
    @match m.agg begin
        :macro => "macro_recall"
        :micro => "micro_recall"
        :nothing => "per_class_recall"
    end
end

initial_state(::Recall{N}) where N = (;tp=zeros(Int, N), fn=zeros(Int, N))

function batch_state(::Recall{N}, y_pred, y_true) where N
    TP, _, _, FN = _tfpn(y_pred, y_true)
    return (;tp=TP, fn=FN)
end

merge_state(::Recall, state1, state2) = (;tp=state1.tp .+ state2.tp, fn=state1.fn .+ state2.fn)

metric_value(x::Recall, state) = _recall(state.tp, state.fn; agg=x.agg)

"""
    BinaryRecall()

A variant of `Recall` specialized for binary classification.
"""
struct BinaryRecall <: ClassificationMetric{2} end

name(::BinaryRecall) = "binary_recall"

initial_state(::BinaryRecall) = (;tp=0, fn=0)

function batch_state(::BinaryRecall, y_pred, y_true)
    TP, _, _, FN = _tfpn(y_pred, y_true)
    return (;tp=TP[2], fn=FN[2])
end

merge_state(::BinaryRecall, state1, state2) = (;tp=state1.tp + state2.tp, fn=state1.fn + state2.fn)

metric_value(::BinaryRecall, state) = (state.tp + eps(Float64)) / (state.tp + state.fn + eps(Float64))

"""
    F1Score(nclasses::Int; agg=:macro)

F1 Score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.

# Parameters
- `nclasses::Int`: The number of classes for the classification task.

# Keyword Parameters
- `agg`: Specifies the type of F1 score aggregation to be computed. The possible values are:
    - `:macro`: Calculates macro-averaged F1 score, which computes the F1 score for each class independently and then takes the average.
    - `:micro`: Calculates micro-averaged F1 score, which aggregates the contributions of all classes to compute a single F1 score.
    - `nothing`: Calculates the per-class F1 score, which is returned as a `Vector` with the same length as `classes`.
"""
struct F1Score{N} <: ClassificationMetric{N}
    agg::Symbol

    function F1Score(nclasses::Int; agg=:macro)
        @argcheck nclasses > 0
        @argcheck agg in (:macro, :micro, nothing)
        new{nclasses}(Symbol(agg))
    end
end

function name(m::F1Score)
    @match m.agg begin
        :macro => "macro_f1_score"
        :micro => "micro_f1_score"
        :nothing => "per_class_f1_score"
    end
end

initial_state(::F1Score{N}) where N = (;tp=zeros(Int, N), fp=zeros(Int, N), fn=zeros(Int, N))

function batch_state(::F1Score{N}, y_pred, y_true) where N
    TP, _, FP, FN = _tfpn(y_pred, y_true)
    return (;tp=TP, fp=FP, fn=FN)
end

function merge_state(::F1Score, state1, state2)
    return (;tp=state1.tp .+ state2.tp, fp=state1.fp .+ state2.fp, fn=state1.fn .+ state2.fn)
end

function metric_value(x::F1Score, state)
    precision = _precision(state.tp, state.fp; x.agg)
    recall = _recall(state.tp, state.fn; x.agg)
    return (2 .* precision .* recall) ./ (precision .+ recall)
end