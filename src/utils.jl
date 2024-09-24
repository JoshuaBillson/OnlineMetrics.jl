# Merge to NamedTuples by applying a binary function to each matching field
function _merge_tuples(op, nt1::T, nt2::T) where {T <: NamedTuple}
    fieldnames = keys(nt1)
    return NamedTuple{fieldnames}(op(nt1[i], nt2[i]) for i in fieldnames)
end

_flatten(x::AbstractArray{<:Real,4}) = @pipe permutedims(x, (3, 1, 2, 4)) |> reshape(_, (size(x, 3), :))

_onecold(x::AbstractArray, dim::Int) = mapslices(argmax, x, dims=dim) .- 1

_classes(nclasses::Int) = collect(0:nclasses-1)

_tp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* y)

_tn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* (1 .- y))

_fp(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum(ŷ .* (1 .- y))

_fn(ŷ::AbstractArray{<:Integer}, y::AbstractArray{<:Integer}) = sum((1 .- ŷ) .* y)