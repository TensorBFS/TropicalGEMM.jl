using TropicalGEMM, Octavian, LoopVectorization
using TropicalNumbers
using Test

function distance(a::AbstractArray{<:Tropical}, b::AbstractArray{<:Tropical})
    sum(abs.(content.(a) .- content.(b)))
end

function naivemm!(o::Matrix, a::Matrix, b::Matrix)
    @assert size(a, 2) == size(b, 1) && size(o) == (size(a, 1), size(b, 2))
    for j=1:size(b, 2)
        for k=1:size(a, 2)
            for i=1:size(a, 1)
                @inbounds o[i,j] += a[i,k] * b[k,j]
            end
        end
    end
    return o
end

@testset "mydot" begin
    function mydot(a,b)
        s = zero(promote_type(eltype(a),eltype(b)))
        @avx for i in eachindex(a,b)
            s += a[i]*b[i]
        end
        s
    end

    a = Tropical.(randn(10))
    b = Tropical.(randn(10))

    @test mydot(a, b) ≈ transpose(a) * b
    @test LoopVectorization.check_args(TropicalF64, TropicalF64)
end


@testset "matmul" begin
    for n in [4, 40]
        a = Tropical.(randn(n, n))
        b = Tropical.(randn(n, n))
        @test distance(Octavian.matmul_serial(a, b), a*b) ≈ 0
        @test distance(Octavian.matmul_serial(a, a), a*a) ≈ 0
        @test distance(Octavian.matmul(a, b), a*b) ≈ 0
    end
end
