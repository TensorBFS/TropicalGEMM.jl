# you will need this to use Tropical numbers in CUDA.
using CUDA
CUDA.allowscalar(false)
using TropicalGEMM: XTranspose, NativeTypes, Tropical, TropicalTypes, CountingTropical
using LinearAlgebra
using Test

const CTranspose{T} = Transpose{T, <:StridedCuVecOrMat{T}}
for TT in [:(Tropical{<:NativeTypes}), :TropicalTypes]
    for RT in [TT, :Real]
        for (TA, CTA) in [(:CuMatrix, :CuMatrix), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}))]
            for (TB, CTB) in [(:CuMatrix, :CuMatrix), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}))]
                @eval function LinearAlgebra.mul!(o::CuMatrix{T}, a::$TA{T}, b::$TB{T}, α::$RT, β::$RT) where {T<:$TT}
                    CUDA.CUBLAS.gemm_dispatch!(o, a, b, α, β)
                end
            end
        end
    end
end

@testset "cuda patch" begin
    for T in [Tropical{Float64}, CountingTropical{Float64,Float64}]
        a = T.(CUDA.randn(4, 4))
        b = T.(CUDA.randn(4))
        for A in [transpose(a), a, transpose(b)]
            for B in [transpose(a), a, b]
                if !(size(A) == (1,4) && size(B) == (4,))
                    res0 = Array(A) * Array(B)
                    res1 = A * B
                    res2 = mul!(CUDA.zeros(T, size(res0)...), A, B, true, false)
                    @test Array(res1) ≈ res0
                    @test Array(res2) ≈ res0
                end
            end
        end
    end
end