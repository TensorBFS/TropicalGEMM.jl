using LinearAlgebra, Octavian, Random

@testset "Reject conjugating adjoints before pointer kernels" begin
    rng = MersenneTwister(19)
    for F in (Float32, Float64)
        T = Tropical{F}
        A = T.(rand(rng, F, 7, 7))
        B = T.(rand(rng, F, 7, 7))
        C = fill(T(F(-3)), 7, 7)
        original = copy(C)
        for (c, a, b) in ((C, A', B), (C, A, B'), (C, A', B'),
                          (C', A, B), (C, view(A', :, :), B),
                          (C, A, view(B', :, :)), (view(C', :, :), A, B))
            for f in (matmul!, matmul_serial!), nargs in 3:5
                args = (c, a, b, one(T), zero(T))
                @test_throws ArgumentError f(args[1:nargs]...)
                @test C == original
            end
            @test_throws ArgumentError matmul!(c, a, b, one(T), zero(T), 1)
            @test C == original
        end
        for f in (matmul!, matmul_serial!)
            x = copy(A[:, 1]); y = similar(x)
            @test_throws ArgumentError f(y, A', x)
        end
    end
end

@testset "Nonconjugating transpose retains tropical products" begin
    rng = MersenneTwister(190)
    for F in (Float32, Float64), n in (7, 40, 129)
        d = rand(rng, F, n, n)
        A = Tropical.(d)
        expected = Tropical.([maximum(d[k,j] + d[i,k] for k in 1:n)
                              for j in 1:n, i in 1:n])
        for f in (matmul!, matmul_serial!)
            C = similar(A)
            @test f(C, transpose(A), transpose(A)) == expected
            @test f(transpose(C), transpose(A), transpose(A)) == expected
        end
        @test mul!(similar(A), transpose(A), transpose(A)) == expected
    end
end
