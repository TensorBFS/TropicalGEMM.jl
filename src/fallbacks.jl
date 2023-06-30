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

# For types not nativelly supported, go to fallback.
# Overwrite the `mul!` in LinearAlgebra (also changes the behavior of `*` in Base)!
function LinearAlgebra.mul!(o::StridedMaybeAdjOrTransMat{TO}, a::StridedMaybeAdjOrTransMat, b::StridedMaybeAdjOrTransMat, α::Number, β::Number) where TO
    α = _convert_to_static(TO, α)
    β = _convert_to_static(TO, β)
    naive_mul!(o, a, b, α, β)
end

Base.:*(a::T, b::StaticInt{0}) where T<:TropicalTypes = zero(T)
Base.:*(a::T, b::StaticInt{1}) where T<:TropicalTypes = a
Base.:*(b::StaticInt{0}, a::T) where T<:TropicalTypes = zero(T)
Base.:*(b::StaticInt{1}, a::T) where T<:TropicalTypes = a