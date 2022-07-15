module TropicalGEMM

using LinearAlgebra, TropicalNumbers, VectorizationBase, LoopVectorization

export Tropical, TropicalF64, TropicalF32, TropicalF16

include("fallbacks.jl")
include("gemm.jl")

if Base.VERSION >= v"1.4.2"
    # precompilation
    for (T1,T2) in [[TropicalF64, TropicalF64], [TropicalF32, TropicalF32], [Tropical{Int64}, Tropical{Int64}]]
        To = promote_type(T1, T2)
        finplace = Octavian.matmul!
        for tA in [true, false]
            TA = tA ? Transpose{T1, Matrix{T1}} : Matrix{T1}
            for tB in [true, false]
                TB = tB ? Transpose{T2, Matrix{T2}} : Matrix{T2}
                precompile(finplace, (Matrix{To},TA, TB,StaticInt{1},StaticInt{0}))
            end
        end
    end
end

end
