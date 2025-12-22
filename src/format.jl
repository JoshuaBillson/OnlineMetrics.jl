"""
    AbstractDataFormat

An abstract type representing a data format for classification tasks.

Subtypes should implement the [`format`](@ref) and [`validate`](@ref) methods.
"""
abstract type AbstractDataFormat end

"""
    format(df::Nothing, x::Array{<:Real})
    format(df::AbstractDataFormat, x::Array{<:Real})

Format the input data `x` according to the specified data format `df`.

# Parameters
- `df`: An instance of a subtype of `AbstractDataFormat` or `nothing`.
- `x`: An array of real-valued data to be formatted.

# Returns
The formatted data.
"""
format(::Nothing, x) = x
format(::AbstractDataFormat, x) = throw(ArgumentError("Formatting not implemented for this data format."))

"""
    validate(df::Nothing, x::Array{<:Real})
    validate(df::AbstractDataFormat, x::Array{<:Real})

Validate that the input data `x` conforms to the specified data format `df`.

Raises an `ArgumentError` if the data does not conform.

# Parameters
- `df`: An instance of a subtype of `AbstractDataFormat` or `nothing`.
- `x`: An array of real-valued data to be validated.
"""
validate(::Nothing, x) = nothing
validate(::AbstractDataFormat, x) = throw(ArgumentError("Validation not implemented for this data format."))

"""
    OneHot(nclasses::Int)

A data format consisting of one-hot encoded class labels for `nclasses` classes.

# Input
- If the input data `x` is an array of shape `(D...,1,N)` or `(N,)`, it is interpreted as class logits in the range `[0, nclasses-1]`.
- If the input data `x` is an array of shape `(D...,nclasses,N)`, it is interpreted as one-hot encoded vectors.

# Output
The output is a `Matrix{Bool}` of shape `(nclasses, N)`.
"""
struct OneHot{N} <: AbstractDataFormat 
    OneHot(nclasses::Int) = new{nclasses}()
end

format(::OneHot{N}, x::AbstractArray{<:Real,D}) where {N,D} = _one_hot(x, N)

function validate(::OneHot{C}, x::AbstractArray{<:Real,N}) where {C,N}
    if N == 1 || size(x,N-1) == 1  # Array of class labels
        all(x -> 0 <= x <= C - 1, x) || throw(ArgumentError("Class labels must be in the range [0, $(C-1)]"))
    else  # Array of one-hot encoded vectors
        size(x,N-1) == C || throw(ArgumentError("Expected one-hot encoded vectors of size $C along dimension $(N-1), got size $(size(x,N-1))"))
        all(isapprox.(sum(x, dims=N-1), 1; atol=1e-5)) || throw(ArgumentError("One-hot encoded vectors must sum to 1 along dimension $(N-1)"))
    end
end

_one_hot(x::AbstractMatrix{Bool}, ::Int) = x

_one_hot(x::AbstractVector{<:Real}, nclasses::Int) = _one_hot(round.(Int, x), nclasses)

function _one_hot(x::AbstractArray{<:Real,N}, nclasses::Int) where N
    @argcheck size(x,N-1) == 1 || size(x,N-1) == nclasses
    if size(x,N-1) == 1  # Array of logits or probabilities
        return _one_hot(reshape(x, :), nclasses)
    else  # Array of one-hot encoded vectors
        indices = reshape(mapslices(argmax, x, dims=N-1), :)
        return _one_hot(indices .- 1, nclasses)
    end
end

function _one_hot(x::AbstractVector{<:Integer}, nclasses::Int)
    dst = zeros(Bool, nclasses, length(x))
    for (i, cls) in enumerate(x)
        dst[cls+1, i] = true
    end
    return dst
end