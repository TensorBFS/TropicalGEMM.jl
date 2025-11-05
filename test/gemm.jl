using TropicalGEMM, Octavian, LoopVectorization
using TropicalNumbers, LinearAlgebra
using VectorizationBase: VecUnroll, Vec, StaticInt
using TropicalGEMM: naive_mul!
using Test

function distance(a::AbstractArray{<:BlasSemiringTypes}, b::AbstractArray{<:BlasSemiringTypes})
    sum(abs.(content.(a) .- content.(b)))
end

_eps(::Type{<:BlasSemiringTypes{T}}) where T = eps(T)
_eps(::Type{<:BlasSemiringTypes{T}}) where T<:Integer = 0
_rand(::Type{T}, args...) where T = T <: BlasSemiringTypes{<:Integer} ? T.(rand(0:10, args...)) : T.(randn(args...))
_rand(::Type{T}, args...) where T <: TropicalMaxMul = T <: BlasSemiringTypes{<:Integer} ? T.(rand(0:10, args...)) : T.(rand(args...))


macro test_close(a, b, atol)
    esc(
        :(@test isapprox(distance($a, $b), 0; atol=$atol))
    )
end

@testset "mul with static ints" begin
    @test StaticInt{0}() * Tropical(3.0) == Tropical(-Inf)
    @test StaticInt{1}() * Tropical(3.0) == Tropical(3.0)
    @test Tropical(3.0) * StaticInt{0}() == Tropical(-Inf)
    @test Tropical(3.0) * StaticInt{1}() == Tropical(3.0)
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
    for (T1,T2) in [[TropicalF64, TropicalF64], [TropicalF32, TropicalF32], [TropicalF64, Tropical{Int64}], [Tropical{Int64}, Tropical{Int64}]]
        To = promote_type(T1, T2)
        atol = sqrt(max(_eps(T1), _eps(T2)))
        for n in [0, 1, 4, 40]
            A = _rand(T1, n, n)
            B = _rand(T2, n, n)
            for (f, finplace) in [(Octavian.matmul_serial, Octavian.matmul_serial!),
                    (Octavian.matmul, Octavian.matmul!)]
                for tA in [true, false]
                    a = tA ? transpose(A) : A
                    @info T1,T2,n,f,tA
                    @test_close f(a, a) naive_mul!(similar(a), a, a) atol
                    for tB in [true, false]
                        b = tB ? transpose(B) : B
                        @info T1,T2,n,f,tA,tB
                        @test_close f(a, b) naive_mul!(similar(a), a, b) atol
                        α, β = _rand(To,2)
                        c = _rand(To, n, n) .|> T1
                        @test_close finplace(copy(c), a, b, α, β) naive_mul!(copy(c), a, b, α, β) atol
                        sa = view(a, 1:min(n, 2), :)
                        sb = view(b, :, 1:min(n,2))
                        c = _rand(To, min(n,2), min(n,2))
                        @test_close finplace(copy(c), sa, sb, α, β) naive_mul!(copy(c), sa, sb, α, β) atol
                    end
                end
            end
        end
    end
end

@testset "*" begin
    a = CountingTropical{Float64}.(randn(5,5))
    b = CountingTropical{Float64}.(randn(5,50))
    @test a * b == naive_mul!(similar(b), a, b)
    a = Tropical{Float64}.(randn(5,5))
    b = Tropical{Float64}.(randn(5,50))
    @test_close a * b naive_mul!(similar(b), a, b) 1e-12
end

@testset "fix julia-1.5" begin
    x=Tropical(Vec(1.0, 2.0))
    @test VecUnroll((x, x)) === Tropical(VecUnroll((Vec(1.0, 2.0), Vec(1.0, 2.0))))
end

@testset "fix nan bug" begin
    res = LinearAlgebra.mul!(Tropical.(fill(NaN, 2, 2)), transpose(Tropical.(randn(2,2))), Tropical.(randn(2,2)), 1, 0)
    @test !any(isnan, res)
end

@testset "MinPlus, MaxMul, MaxMin, and Bitwise" begin
    for (T1,T2) in [
            [TropicalMinPlusF64, TropicalMinPlusF64],
            [TropicalMinPlus{Int64}, TropicalMinPlus{Int64}],
            [TropicalMaxMulF64, TropicalMaxMulF64],
            [TropicalMaxMul{Int64}, TropicalMaxMul{Int64}],
            [TropicalMaxMinF64, TropicalMaxMinF64],
            [TropicalMaxMinI64, TropicalMaxMinI64],
            [TropicalBitwiseI64, TropicalBitwiseI64],
        ]
        for n in [0, 1, 4, 40]
            A = _rand(T1, n, n)
            B = _rand(T2, n, n)
            for (f, finplace) in [(Octavian.matmul_serial, Octavian.matmul_serial!),
                    (Octavian.matmul, Octavian.matmul!)]
                for tA in [true, false]
                    a = tA ? transpose(A) : A
                    @info T1,T2,n,f,tA
                    @test_close f(a, a) naive_mul!(similar(a), a, a) 1e-12
                    for tB in [true, false]
                        b = tB ? transpose(B) : B
                        @info T1,T2,n,f,tA,tB
                        @test_close f(a, b) naive_mul!(similar(a), a, b) 1e-12
                        α, β = _rand(T1,2)
                        c = _rand(T1, n, n)
                        @test_close finplace(copy(c), a, b, α, β) naive_mul!(copy(c), a, b, α, β) 1e-12
                        sa = view(a, 1:min(n, 2), :)
                        sb = view(b, :, 1:min(n,2))
                        c = _rand(T1, min(n,2), min(n,2))
                        @test_close finplace(copy(c), sa, sb, α, β) naive_mul!(copy(c), sa, sb, α, β) 1e-12
                    end
                end
            end
        end
    end
end
