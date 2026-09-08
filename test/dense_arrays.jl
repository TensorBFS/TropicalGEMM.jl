# Exercise the DenseArray interface without requiring a package that raises the
# minimum Julia version of this package's test environment.
struct DenseTestMatrix{T,PointerAccessible} <: DenseMatrix{T}
    data::Matrix{T}
end

DenseTestMatrix(data::Matrix{T}) where T = DenseTestMatrix{T,true}(data)
Base.size(a::DenseTestMatrix) = size(a.data)
Base.getindex(a::DenseTestMatrix, i::Int) = a.data[i]
Base.setindex!(a::DenseTestMatrix, x, i::Int) = (a.data[i] = x)
Base.IndexStyle(::Type{<:DenseTestMatrix}) = IndexLinear()
Base.strides(a::DenseTestMatrix) = strides(a.data)
Base.elsize(::Type{<:DenseTestMatrix{T}}) where T = sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, a::DenseTestMatrix{T,true}) where T = Base.unsafe_convert(Ptr{T}, a.data)
Octavian.ArrayInterface.device(::Type{<:DenseTestMatrix{T,false}}) where T = Octavian.ArrayInterface.CPUIndex()

@testset "Dense matrix multiplication dispatch" begin
    for T in (TropicalF32, TropicalMinPlusF64), n in (1, 4, 17)
        a = DenseTestMatrix(T.(rand(n, n)))
        b = DenseTestMatrix(T.(rand(n, n)))
        for wrap in (identity, transpose, x -> view(x, :, :)),
                (α, β) in ((one(T), zero(T)), (T(0.5), T(0.25)))
            aa, bb = wrap(a), wrap(b)
            c = DenseTestMatrix(T.(rand(n, n)))
            expected = naive_mul!(copy(c.data), aa, bb, α, β)
            @test which(mul!, Tuple{typeof(c), typeof(aa), typeof(bb), T, T}).module === TropicalGEMM
            @test content.(mul!(c, aa, bb, α, β)) ≈ content.(expected)
        end
        # A dense container without CPU pointer access must retain the generic
        # algorithm. Its unsafe_convert is deliberately not implemented.
        c = DenseTestMatrix{T,false}(T.(rand(n, n)))
        expected = naive_mul!(copy(c.data), a, b, one(T), one(T))
        @test mul!(c, a, b, one(T), one(T)) == expected
    end
end
