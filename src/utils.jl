# Merge to NamedTuples by applying a binary function to each matching field
function _merge_tuples(op, nt1::T, nt2::T) where {T <: NamedTuple}
    fieldnames = keys(nt1)
    return NamedTuple{fieldnames}(op(nt1[i], nt2[i]) for i in fieldnames)
end

_flatten(x::AbstractArray{<:Real,4}) = @pipe permutedims(x, (3, 1, 2, 4)) |> reshape(_, (size(x, 3), :))

_onecold(x::AbstractArray, dim::Int) = mapslices(argmax, x, dims=dim) .- 1

_onehot(x::AbstractVector, labels) = _onehot(reshape(x, (1,:)), labels)
function _onehot(x::AbstractArray{<:Any,N}, labels) where {N}
    if size(x, N-1) == 1
        return cat(map(label -> x .== label, labels)..., dims=N-1)
    end
    return x
end

_classes(nclasses::Int) = collect(0:nclasses-1)

_round(x::Integer, ::Int) = x
_round(x::Real, digits::Int) = round(x, digits=digits)
_round(x::AbstractArray, digits::Int) = _round.(x, digits)

_tp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* y)

_tn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* (1 .- y))

_fp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* (1 .- y))

_fn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* y)

_largest_n(x::AbstractVector{<:Real}, n::Int) = partialsortperm(x, 1:n, rev=true)
function _largest_n(x::AbstractArray{<:Real,N}, n::Int) where N
    mapslices(x -> _largest_n(x, n), x, dims=N-1)
end