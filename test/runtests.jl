using OnlineMetrics
using Test

function evaluate_metric(metric, ŷ, y)
    step!(metric, ŷ, y)
    return value(metric)
end

function evaluate_metric_batched(metric, ŷ, y)
    for (pred, label) in zip(generate_batches(ŷ), generate_batches(y))
        step!(metric, pred, label)
    end
    return value(metric)
end

generate_batches(x::AbstractVector) = [x[i:i] for i in eachindex(x)]
function generate_batches(x::AbstractArray{<:Any,N}) where N
    obs = size(x, N)
    return [selectdim(x, N, i:i) for i in 1:obs]
end

hard_labels(x::AbstractVector{<:AbstractFloat}) = round.(Int, x)

onehot_labels(x::AbstractVector{<:AbstractFloat}) = OnlineMetrics._onehot(round.(Int, x), 0:1)

function softmax_labels(x::AbstractVector{<:AbstractFloat})
    return cat(1 .- reshape(x, (1,:)), reshape(x, (1,:)), dims=1)
end

@testset "class metrics" begin
    y_pred = [0.1, 0.8, 0.51, 0.49]
    y_correct = [0.05, 0.95, 0.95, 0.05]
    y_incorrect = [0.95, 0.05, 0.05, 0.95]
    y_mixed = [0.05, 0.05, 0.95, 0.95]

    # Accuracy - All Correct
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_correct, hard_labels(y_correct), onehot_labels(y_correct), softmax_labels(y_correct))
            @test evaluate_metric(Accuracy(), ŷ, y) == 1
            @test evaluate_metric_batched(Accuracy(), ŷ, y) == 1
        end
    end

    # Accuracy - All Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_incorrect, hard_labels(y_incorrect), onehot_labels(y_incorrect), softmax_labels(y_incorrect))
            @test evaluate_metric(Accuracy(), ŷ, y) == 0
            @test evaluate_metric_batched(Accuracy(), ŷ, y) == 0
        end
    end

    # Accuracy - Half Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_mixed, hard_labels(y_mixed), onehot_labels(y_mixed), softmax_labels(y_mixed))
            @test evaluate_metric(Accuracy(), ŷ, y) == 0.5
            @test evaluate_metric_batched(Accuracy(), ŷ, y) == 0.5
        end
    end

    # MIoU - All Correct
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_correct, hard_labels(y_correct), onehot_labels(y_correct), softmax_labels(y_correct))
            @test evaluate_metric(MIoU(2), ŷ, y) == 1
            @test evaluate_metric_batched(MIoU(2), ŷ, y) == 1
        end
    end

    # MIoU - All Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_incorrect, hard_labels(y_incorrect), onehot_labels(y_incorrect), softmax_labels(y_incorrect))
            @test evaluate_metric(MIoU(2), ŷ, y) == 0
            @test evaluate_metric_batched(MIoU(2), ŷ, y) == 0
        end
    end

    # MIoU - Half Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_mixed, hard_labels(y_mixed), onehot_labels(y_mixed), softmax_labels(y_mixed))
            @test evaluate_metric(MIoU(2), ŷ, y) ≈ 1/3
            @test evaluate_metric_batched(MIoU(2), ŷ, y) ≈ 1/3
        end
    end

    # MIoU - No Positive Labels
    @test evaluate_metric(MIoU(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(MIoU(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Precision - All Correct
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_correct, hard_labels(y_correct), onehot_labels(y_correct), softmax_labels(y_correct))
            @test evaluate_metric(Precision(2), ŷ, y) == 1
            @test evaluate_metric_batched(Precision(2), ŷ, y) == 1
        end
    end

    # Precision - All Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_incorrect, hard_labels(y_incorrect), onehot_labels(y_incorrect), softmax_labels(y_incorrect))
            @test evaluate_metric(Precision(2), ŷ, y) == 0
            @test evaluate_metric_batched(Precision(2), ŷ, y) == 0
        end
    end

    # Precision - Half Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_mixed, hard_labels(y_mixed), onehot_labels(y_mixed), softmax_labels(y_mixed))
            @test evaluate_metric(Precision(2), ŷ, y) == 0.5
            @test evaluate_metric_batched(Precision(2), ŷ, y) == 0.5
        end
    end

    # Precision - No Positive Labels
    @test evaluate_metric(Precision(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(Precision(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Recall - All Correct
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_correct, hard_labels(y_correct), onehot_labels(y_correct), softmax_labels(y_correct))
            @test evaluate_metric(Recall(2), ŷ, y) == 1
            @test evaluate_metric_batched(Recall(2), ŷ, y) == 1
        end
    end

    # Recall - All Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_incorrect, hard_labels(y_incorrect), onehot_labels(y_incorrect), softmax_labels(y_incorrect))
            @test evaluate_metric(Recall(2), ŷ, y) == 0
            @test evaluate_metric_batched(Recall(2), ŷ, y) == 0
        end
    end

    # Recall - Half Incorrect
    for ŷ in (y_pred, hard_labels(y_pred), onehot_labels(y_pred), softmax_labels(y_pred))
        for y in (y_mixed, hard_labels(y_mixed), onehot_labels(y_mixed), softmax_labels(y_mixed))
            @test evaluate_metric(Recall(2), ŷ, y) == 0.5
            @test evaluate_metric_batched(Recall(2), ŷ, y) == 0.5
        end
    end

    # Recall - No Positive Labels
    @test evaluate_metric(Recall(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(Recall(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Confusion Matrix
    evaluate_metric(ConfusionMatrix(2), y_pred, y_correct) == [2 0; 0 2]
    evaluate_metric(ConfusionMatrix(2), y_pred, y_incorrect) == [0 2; 2 0]
    evaluate_metric(ConfusionMatrix(2), y_pred, y_mixed) == [1 1; 1 1]
end
