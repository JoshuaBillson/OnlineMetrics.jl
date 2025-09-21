one_hot(x::OneHotMatrix, ::Int) = x
one_hot(x::AbstractVector{<:Integer}, nclasses::Int) = onehotbatch(x, 0:nclasses-1)
one_hot(x::AbstractVector{<:Real}, nclasses::Int) = one_hot(round.(Int, x), nclasses)
function one_hot(x::AbstractArray{<:Real,N}, nclasses::Int) where N
    if size(x,N-1) == 1
        return one_hot(reshape(x, :), nclasses)
    elseif size(x,N-1) == nclasses
        return @pipe mapslices(argmax, x, dims=N-1) |> reshape(_, :) |> one_hot(_ .- 1, nclasses)
    else
        throw(ArgumentError("Expected dim $(N-1) to be of size 1 of $nclasses, got $(size(x,N-1))"))
    end
end

function weighted_average(x::AbstractArray{<:Real}, avg::Real, n::Integer)
    m = length(x)
    return (avg * (n / (n + m))) + (mean(x) * (m / (n + m)))
end

_tp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* y)

_tn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* (1 .- y))

_fp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* (1 .- y))

_fn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* y)

function _confusion_matrix(y_pred::OneHotMatrix, y_true::OneHotMatrix)
    @assert size(y_pred) == size(y_true)
    return y_pred * transpose(y_true) 
end

_tfpn(y_pred::OneHotMatrix, y_true::OneHotMatrix) = _confusion_matrix(y_pred, y_true) |> _tfpn
function _tfpn(confusion_matrix::AbstractMatrix)
    nclasses = size(confusion_matrix, 1)
    TP = zeros(Int, nclasses)
    TN = zeros(Int, nclasses)
    FP = zeros(Int, nclasses)
    FN = zeros(Int, nclasses)
    for c in 1:nclasses
        TP[c] = confusion_matrix[c,c]
        FP[c] = sum(confusion_matrix[:,c]) - TP[c]
        FN[c] = sum(confusion_matrix[c,:]) - TP[c]
        TN[c] = sum(confusion_matrix) - TP[c] - FP[c] - FN[c]
    end
    return TP, TN, FP, FN
end

_largest_n(x::AbstractVector{<:Real}, n::Int) = sortperm(x, rev=true)
function _largest_n(x::AbstractArray{<:Real,N}, n::Int) where N
    mapslices(x -> _largest_n(x, n), x, dims=N-1)
end