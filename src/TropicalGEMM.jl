module TropicalGEMM

using LinearAlgebra, TropicalNumbers, VectorizationBase, LoopVectorization
using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer, AbstractSIMD
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference,
    vmaximum, fmap, FloatingTypes, IntegerIndex, LazyMulAdd
using LinearAlgebra: StridedMaybeAdjOrTransMat

export Tropical, TropicalF64, TropicalF32, TropicalF16

include("fallbacks.jl")
include("gemm.jl")

end
