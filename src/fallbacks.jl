# change the loop order in Base.
function naive_mul!(o::AbstractMatrix{T0}, a::AbstractMatrix{T1}, b::AbstractMatrix{T2}, α=one(T0), β=zero(T0)) where {T0,T1,T2}
    @assert size(a, 2) == size(b, 1) && size(o) == (size(a, 1), size(b, 2))
    a = convert(Matrix, a)
    b = convert(Matrix, b)
    for j=1:size(b, 2)
        if !iszero(β)
            @inbounds for i=1:size(a, 1)
                o[i,j] = β * o[i,j]
            end
        else
            @inbounds for i=1:size(a, 1)
                o[i,j] = zero(T0)
            end
        end
        for k=1:size(a, 2)
            for i=1:size(a, 1)
                @inbounds o[i,j] += α * a[i,k] * b[k,j]
            end
        end
    end
    return o
end

# Overwrite the `mul!` in LinearAlgebra (also changes the behavior of `*` in Base)!
for TA in [:(AbstractMatrix{T} where T<:TropicalTypes), :(Transpose{T,S} where {T<:TropicalTypes,S<:AbstractVecOrMat{T}})]
    for TB in [:(AbstractMatrix{T} where T<:TropicalTypes), :(Transpose{T,S} where {T<:TropicalTypes,S<:AbstractVecOrMat{T}})]
        @eval @inline function LinearAlgebra.mul!(o::AbstractMatrix{TO}, a::$TA, b::$TB, α::Number, β::Number) where TO
            α = _convert_to_tropical(TO, α)
            β = _convert_to_tropical(TO, β)
            naive_mul!(o, a, b, α, β)
        end
    end
end

_convert_to_tropical(::Type{T}, α::TropicalTypes) where T<:TropicalTypes = α
function _convert_to_tropical(::Type{T}, α::Number) where T<:TropicalTypes
    if iszero(α)
        return zero(T)
    elseif isone(α)
        return one(T)
    else
        throw(ArgumentError("$α is not a valid tropical number."))
    end
end