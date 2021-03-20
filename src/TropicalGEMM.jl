module TropicalGEMM

using LinearAlgebra, TropicalNumbers, VectorizationBase, LoopVectorization

export Tropical, TropicalF64, TropicalF32, TropicalF16

include("fallbacks.jl")
include("gemm.jl")

end
