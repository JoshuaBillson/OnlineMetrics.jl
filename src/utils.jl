# Merge to NamedTuples by applying a binary function to each matching field
function _merge_tuples(op, nt1::T, nt2::T) where {T <: NamedTuple}
    fieldnames = keys(nt1)
    return NamedTuple{fieldnames}(op(nt1[i], nt2[i]) for i in fieldnames)
end

_flatten(x::AbstractArray{<:Real,4}) = @pipe permutedims(x, (3, 1, 2, 4)) |> reshape(_, (size(x, 3), :))

_logits(x::AbstractVector{<:Integer}) = x
_logits(x::AbstractVector{<:Real}) = round.(Int, x)
function _logits(x::AbstractArray{<:Real,N}) where N
    if size(x, N-1) > 1
        return vec(mapslices(argmax, x, dims=N-1) .- 1)
    else
        return round.(Int, x) |> vec
    end

end

_onehot(x::AbstractVector, labels) = _onehot(reshape(x, (1,:)), labels)
function _onehot(x::AbstractArray{T,N}, labels) where {T<:Real,N}
    if size(x, N-1) == 1  # Labels Encoded as Logits
        return cat(map(label -> x .== label, labels)..., dims=N-1)
    else  # Labels Encoded as Smooth One-Hot
        @argcheck length(labels) == size(x,N-1)
        dst = zeros(T, size(x))
        dst[argmax(x, dims=N-1)] .= T(1)
        return dst
    end
end

_classes(nclasses::Int) = collect(0:nclasses-1)

_round(x::Integer, ::Int) = x
_round(x::Real, digits::Int) = round(x, digits=digits)
_round(x::AbstractArray, digits::Int) = _round.(x, digits)

_tp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* y)

_tn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* (1 .- y))

_fp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* (1 .- y))

_fn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* y)

function _tfpn(confusion_matrix::AbstractMatrix{<:Real})
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