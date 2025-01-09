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

@testset "class metrics" begin
    y_pred_soft = [0.1, 0.8, 0.51, 0.49]
    y_pred_hard = [0, 1, 1, 0]
    y_pred_onehot = hcat([1, 0], [0, 1], [0, 1], [1, 0])
    y_pred_softmax = hcat([0.9, 0.1], [0.2, 0.8], [0.49, 0.51], [0.51, 0.49])
    y_correct = [0, 1, 1, 0]
    y_incorrect = [1, 0, 0, 1]
    y_mixed = [0, 0, 1, 1]

    # Accuracy - Soft Predictions
    @test evaluate_metric(Accuracy(), y_pred_soft, y_correct) == 1
    @test evaluate_metric(Accuracy(), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric(Accuracy(), y_pred_soft, y_mixed) == 0.5

    # Accuracy - Multi Batch
    @test evaluate_metric_batched(Accuracy(), y_pred_soft, y_correct) == 1
    @test evaluate_metric_batched(Accuracy(), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric_batched(Accuracy(), y_pred_soft, y_mixed) == 0.5

    # Accuracy - Hard Predictions
    @test evaluate_metric(Accuracy(), y_pred_hard, y_correct) == 1
    @test evaluate_metric(Accuracy(), y_pred_hard, y_incorrect) == 0
    @test evaluate_metric(Accuracy(), y_pred_hard, y_mixed) == 0.5

    # Accuracy - One-Hot Predictions
    @test evaluate_metric(Accuracy(), y_pred_onehot, y_correct) == 1
    @test evaluate_metric(Accuracy(), y_pred_onehot, y_incorrect) == 0
    @test evaluate_metric(Accuracy(), y_pred_onehot, y_mixed) == 0.5

    # Accuracy - Softmax Predictions
    @test evaluate_metric(Accuracy(), y_pred_softmax, y_correct) == 1
    @test evaluate_metric(Accuracy(), y_pred_softmax, y_incorrect) == 0
    @test evaluate_metric(Accuracy(), y_pred_softmax, y_mixed) == 0.5

    # Accuracy - Soft Labels
    evaluate_metric(Accuracy(), y_pred_hard, [0.05, 0.05, 0.95, 0.95]) == 0.5

    # Accuracy - One-Hot Labels
    evaluate_metric(Accuracy(), y_pred_softmax, hcat([1, 0], [1, 0], [0, 1], [0, 1])) == 0.5

    # MIoU - Soft Predictions
    @test evaluate_metric(MIoU(2), y_pred_soft, y_correct) == 1
    @test isapprox(evaluate_metric(MIoU(2), y_pred_soft, y_incorrect), 0, atol=1e-15)
    @test evaluate_metric(MIoU(2), y_pred_soft, y_mixed) ≈ 1 / 3

    # MIoU - Multi Batch
    @test evaluate_metric_batched(MIoU(2), y_pred_soft, y_correct) == 1
    @test isapprox(evaluate_metric_batched(MIoU(2), y_pred_soft, y_incorrect), 0, atol=1e-15)
    @test evaluate_metric_batched(MIoU(2), y_pred_soft, y_mixed) ≈ 1/3

    # MIoU - Hard Predictions
    @test evaluate_metric_batched(MIoU(2), y_pred_hard, y_correct) == 1
    @test isapprox(evaluate_metric_batched(MIoU(2), y_pred_hard, y_incorrect), 0, atol=1e-15)
    @test evaluate_metric_batched(MIoU(2), y_pred_hard, y_mixed) ≈ 1/3

    # MIoU - One-Hot Predictions
    @test evaluate_metric_batched(MIoU(2), y_pred_onehot, y_correct) == 1
    @test isapprox(evaluate_metric_batched(MIoU(2), y_pred_onehot, y_incorrect), 0, atol=1e-15)
    @test evaluate_metric_batched(MIoU(2), y_pred_onehot, y_mixed) ≈ 1/3

    # MIoU - Softmax Predictions
    @test evaluate_metric_batched(MIoU(2), y_pred_softmax, y_correct) == 1
    @test isapprox(evaluate_metric_batched(MIoU(2), y_pred_softmax, y_incorrect), 0, atol=1e-15)
    @test evaluate_metric_batched(MIoU(2), y_pred_softmax, y_mixed) ≈ 1/3

    # MIoU - No Positive Labels
    @test evaluate_metric(MIoU(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(MIoU(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Precision - Soft Predictions
    @test evaluate_metric(Precision(2), y_pred_soft, y_correct) == 1
    @test evaluate_metric(Precision(2), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric(Precision(2), y_pred_soft, y_mixed) == 0.5

    # Precision - Multi Batch
    @test evaluate_metric_batched(Precision(2), y_pred_soft, y_correct) == 1
    @test evaluate_metric_batched(Precision(2), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric_batched(Precision(2), y_pred_soft, y_mixed) == 0.5

    # Precision - Hard Predictions
    @test evaluate_metric(Precision(2), y_pred_hard, y_correct) == 1
    @test evaluate_metric(Precision(2), y_pred_hard, y_incorrect) == 0
    @test evaluate_metric(Precision(2), y_pred_hard, y_mixed) == 0.5

    # Precision - One-Hot Predictions
    @test evaluate_metric(Precision(2), y_pred_onehot, y_correct) == 1
    @test evaluate_metric(Precision(2), y_pred_onehot, y_incorrect) == 0
    @test evaluate_metric(Precision(2), y_pred_onehot, y_mixed) == 0.5

    # Precision - Softmax Predictions
    @test evaluate_metric(Precision(2), y_pred_softmax, y_correct) == 1
    @test evaluate_metric(Precision(2), y_pred_softmax, y_incorrect) == 0
    @test evaluate_metric(Precision(2), y_pred_softmax, y_mixed) == 0.5

    # Precision - No Positive Labels
    @test evaluate_metric(Precision(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(Precision(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Precision - Soft Predictions
    @test evaluate_metric(Recall(2), y_pred_soft, y_correct) == 1
    @test evaluate_metric(Recall(2), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric(Recall(2), y_pred_soft, y_mixed) == 0.5

    # Precision - Multi Batch
    @test evaluate_metric_batched(Recall(2), y_pred_soft, y_correct) == 1
    @test evaluate_metric_batched(Recall(2), y_pred_soft, y_incorrect) == 0
    @test evaluate_metric_batched(Recall(2), y_pred_soft, y_mixed) == 0.5

    # Precision - Hard Predictions
    @test evaluate_metric(Recall(2), y_pred_hard, y_correct) == 1
    @test evaluate_metric(Recall(2), y_pred_hard, y_incorrect) == 0
    @test evaluate_metric(Recall(2), y_pred_hard, y_mixed) == 0.5

    # Precision - One-Hot Predictions
    @test evaluate_metric(Recall(2), y_pred_onehot, y_correct) == 1
    @test evaluate_metric(Recall(2), y_pred_onehot, y_incorrect) == 0
    @test evaluate_metric(Recall(2), y_pred_onehot, y_mixed) == 0.5

    # Precision - Softmax Predictions
    @test evaluate_metric(Recall(2), y_pred_softmax, y_correct) == 1
    @test evaluate_metric(Recall(2), y_pred_softmax, y_incorrect) == 0
    @test evaluate_metric(Recall(2), y_pred_softmax, y_mixed) == 0.5

    # Precision - No Positive Labels
    @test evaluate_metric(Recall(2), [0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    @test evaluate_metric(Recall(2), [1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    # Confusion Matrix
    evaluate_metric(ConfusionMatrix(2), y_pred_soft, y_correct) == [2 0; 0 2]
    evaluate_metric(ConfusionMatrix(2), y_pred_soft, y_incorrect) == [0 2; 2 0]
    evaluate_metric(ConfusionMatrix(2), y_pred_soft, y_mixed) == [1 1; 1 1]
end
