module TropicalGEMM

using LinearAlgebra, TropicalNumbers, VectorizationBase, LoopVectorization
using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer, AbstractSIMD
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference,
    vmaximum, fmap, FloatingTypes, IntegerIndex, LazyMulAdd

export Tropical, TropicalF64, TropicalF32

include("fallbacks.jl")
include("gemm.jl")

import PrecompileTools
PrecompileTools.@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    PrecompileTools.@compile_workload begin
        for T in (Float32, Float64, Int64)
            A = Tropical.(rand(T, 10, 10))
            O = Tropical.(rand(T, 10, 10))
            LinearAlgebra.mul!(O, A, A)
        end
    end
end

end
