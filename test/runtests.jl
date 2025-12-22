using OnlineMetrics
using Test

generate_batches(x::Tuple) = zip(map(generate_batches, x)...) |> collect
generate_batches(x::AbstractVector) = [x[i:i] for i in eachindex(x)]
function generate_batches(x::AbstractArray{<:Any,N}) where N
    obs = size(x, N)
    return [selectdim(x, N, i:i) for i in 1:obs]
end

function onehot_labels(x::AbstractVector{<:Integer}, nclasses::Int)
    dst = zeros(Bool, nclasses, length(x))
    for i in eachindex(x)
        dst[x[i]+1, i] = true
    end
    return dst
end

function softmax_labels(x::AbstractVector{<:Integer}, nclasses::Int, smoothing::Float64=0.1)
    hard_labels = onehot_labels(x, nclasses) .|> Float64
    smooth_labels = ((1 - smoothing) .* hard_labels) .+ (smoothing / nclasses)
    return smooth_labels
end

function evaluate_metric(metric::AbstractMetric, y_pred::Vector{Int}, y_true::Vector{Int}, nclasses, expected_value, atol=1e-8)
    for ŷ in [y_pred, onehot_labels(y_pred, nclasses), softmax_labels(y_pred, nclasses)]
        for y in [y_true, onehot_labels(y_true, nclasses), softmax_labels(y_true, nclasses)]
            m = Metric(metric)
            foreach(generate_batches((ŷ, y))) do (ŷ_batch, y_batch)
                step!(m, ŷ_batch, y_batch)
            end
            @test ≈(value(m), expected_value, atol=atol)
        end
    end
end

@testset "data format" begin
    # One Hot
    df = OneHot(2)
    @test format(df, [0,1,0,1]) == [1 0 1 0; 0 1 0 1]
    @test format(df, [0.0,1.0,0.0,1.0]) == [1 0 1 0; 0 1 0 1]
    @test format(df, [0.9 0.25 0.56 0.1; 0.15 0.99 0.01 0.51]) == [1 0 1 0; 0 1 0 1]
end

@testset "utils" begin
    # TPFN 
    y_pred = onehot_labels(rand(Bool, 1000), 2)
    y_true = onehot_labels(rand(Bool, 1000), 2)
    TP, TN, FP, FN = OnlineMetrics._tfpn(y_pred, y_true)
    @test sum(y_pred .&& y_true, dims=2)[:] == TP
    @test sum(.!y_pred .&& .!y_true, dims=2)[:] == TN
    @test sum(y_pred .&& .!y_true, dims=2)[:] == FP
    @test sum(.!y_pred .&& y_true, dims=2)[:] == FN
end

@testset "classification metrics" begin
    # Accuracy
    evaluate_metric(Accuracy(2), [0, 1, 1, 0], [0, 1, 1, 0], 2, 1)  # all correct
    evaluate_metric(Accuracy(2), [0, 1, 1, 0], [1, 0, 0, 1], 2, 0)  # all incorrect
    evaluate_metric(Accuracy(2), [0, 1, 1, 0], [0, 0, 1, 1], 2, 0.5)  # half incorrect
    evaluate_metric(Accuracy(2), [0, 0, 0, 0], [0, 0, 0, 0], 2, 1.0)  # no positive labels
    evaluate_metric(Accuracy(2), [1, 1, 1, 1], [1, 1, 1, 1], 2, 1.0)  # no negative labels

    # mIoU
    evaluate_metric(mIoU(2), [0, 1, 1, 0], [0, 1, 1, 0], 2, 1)  # all correct
    evaluate_metric(mIoU(2), [0, 1, 1, 0], [1, 0, 0, 1], 2, 0)  # all incorrect
    evaluate_metric(mIoU(2), [0, 1, 1, 0], [0, 0, 1, 1], 2, 1/3)  # half incorrect
    evaluate_metric(mIoU(2), [0, 0, 0, 0], [0, 0, 0, 0], 2, 1.0)  # no positive labels
    evaluate_metric(mIoU(2), [1, 1, 1, 1], [1, 1, 1, 1], 2, 1.0)  # no negative labels

    # Precision
    evaluate_metric(Precision(2), [0, 1, 1, 0], [0, 1, 1, 0], 2, 1)  # all correct
    evaluate_metric(Precision(2), [0, 1, 1, 0], [1, 0, 0, 1], 2, 0)  # all incorrect
    evaluate_metric(Precision(2), [0, 1, 1, 0], [0, 0, 1, 1], 2, 0.5)  # half incorrect
    evaluate_metric(Precision(2), [0, 0, 0, 0], [0, 0, 0, 0], 2, 1.0)  # no positive labels
    evaluate_metric(Precision(2), [1, 1, 1, 1], [1, 1, 1, 1], 2, 1.0)  # no negative labels

    # Recall
    evaluate_metric(Recall(2), [0, 1, 1, 0], [0, 1, 1, 0], 2, 1)  # all correct
    evaluate_metric(Recall(2), [0, 1, 1, 0], [1, 0, 0, 1], 2, 0)  # all incorrect
    evaluate_metric(Recall(2), [0, 1, 1, 0], [0, 0, 1, 1], 2, 0.5)  # half incorrect
    evaluate_metric(Recall(2), [0, 0, 0, 0], [0, 0, 0, 0], 2, 1.0)  # no positive labels
    evaluate_metric(Recall(2), [1, 1, 1, 1], [1, 1, 1, 1], 2, 1.0)  # no negative labels
end
