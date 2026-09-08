using TropicalGEMM
using Test

@testset "TropicalGEMM.jl" begin
    include("gemm.jl")
    include("dense_arrays.jl")
    include("adjoints.jl")
end
