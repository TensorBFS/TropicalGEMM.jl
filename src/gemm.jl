using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer, AbstractSIMD
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference,
    vmaximum, fmap, FloatingTypes, IntegerIndex, LazyMulAdd

LoopVectorization.check_args(::Type{T}, ::Type{T}) where T<:Tropical = true
LoopVectorization.check_type(::Type{Tropical{T}}) where {T} = LoopVectorization.check_type(T)

@inline Base.FastMath.add_fast(a::Tropical, b::Tropical) = Tropical(Base.FastMath.max_fast(content(a), content(b)))

@inline function VectorizationBase._vstore!(
    ptr::AbstractStridedPointer, vu::Tropical{<:VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
    VectorizationBase._vstore!(notropical(ptr), content(vu), u, a, s, nt, si)
end

@inline function VectorizationBase._vstore!(ptr::AbstractStridedPointer, vu::Tropical{<:VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
    VectorizationBase._vstore!(notropical(ptr), content(vu), u, m, a, s, nt, si)
end

@inline function VectorizationBase._vstore!(
    g::G, ptr::AbstractStridedPointer{T,D,C}, vu::Tropical{<:VecUnroll{U,W}}, u::Unroll{AU,F,N,AV,1,M,X,I}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X}
    VectorizationBase._vstore!(g, notropical(ptr), content(vu), u, a, s, nt, si)
end
@inline function VectorizationBase.__vstore!(
    f::F, ptr::Ptr{Tropical{T}}, v::Tropical{T}, i::IntegerIndex, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {T<:NativeTypesExceptBit, F<:Function,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
    VectorizationBase.__vstore!(f, Ptr{T}(ptr), content(v), i, a, s, nt, si)
end
@inline function VectorizationBase.__vstore!(
        ptr::Ptr{Tropical{T}}, v::Tropical{VT}, i::I, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {W, T <: NativeTypesExceptBit, VT <: Vec, I <: Index, A <: StaticBool, S <: StaticBool, NT <: StaticBool, RS}
    VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, m, a, s, nt, si)
end
@inline function VectorizationBase.__vstore!(
    ptr::Ptr{Tropical{T}}, v::Tropical{VT}, i::VectorIndex{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {T,VT<:Vec,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS}
    VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, a, s, nt, si)
end

@inline function VectorizationBase.__vload(ptr::Ptr{Tropical{T}}, i::I, m::AbstractMask, a::A, si::StaticInt{RS})  where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
    Tropical(VectorizationBase.__vload(Ptr{T}(ptr), i, m, a, si))
end
@inline function VectorizationBase.__vload(ptr::Ptr{Tropical{T}}, i::I, a::A, si::StaticInt{RS}) where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
    Tropical(VectorizationBase.__vload(Ptr{T}(ptr), i, a, si))
end

@inline function VectorizationBase._vbroadcast(a::Union{Val{W},StaticInt{W}}, s::Tropical{T}, si::StaticInt{RS}) where {W,T,RS}
    Tropical(VectorizationBase._vbroadcast(a, content(s), si))
end

@inline function notropical(ptr::VectorizationBase.StridedPointer{Tropical{T},N,C,B,R,X,O}) where {T,N,C,B,R,X,O}
    stridedpointer(Ptr{T}(ptr.p), ptr.si, StaticInt{B}())
end

@inline function notropical(ptr::OffsetPrecalc{<:Tropical})
    VectorizationBase.OffsetPrecalc(notropical(ptr.ptr), ptr.precalc)
end

@inline function VectorizationBase._vload(ptr::AbstractStridedPointer{Tropical{T}}, u::Unroll, ::A, ::StaticInt{RS}) where {T,A<:StaticBool,RS}
    res = VectorizationBase._vload(notropical(ptr), u, A(), StaticInt{RS}())
    Tropical(res)
end

@inline function VectorizationBase._vload(ptr::AbstractStridedPointer{Tropical{T}}, u::Unroll, m::AbstractMask, ::A, ::StaticInt{RS}) where {T,A<:StaticBool,RS}
    res = VectorizationBase._vload(notropical(ptr), u, m, A(), StaticInt{RS}())
    Tropical(res)
end

@generated function VectorizationBase.zero_vecunroll(::StaticInt{N}, ::StaticInt{W}, ::Type{Tropical{T}}, ::StaticInt{RS}) where {N,W,T,RS}
    quote
        $(Expr(:meta,:inline))
        t = Base.Cartesian.@ntuple $N n -> VectorizationBase._vbroadcast(StaticInt{$W}(), $(zero(Tropical{T}).n), StaticInt{$RS}())
        Tropical(VecUnroll(t))
    end
end

@inline function Base.promote_rule(::Type{Tropical{T1}}, ::Type{Tropical{T2}}) where {T1<:VecUnroll,T2<:Vec}
    Tropical{promote_rule(T1, T2)}
end

@inline function VectorizationBase._vzero(::StaticInt{W}, ::Type{T}, ::StaticInt{RS}) where {W,FT,T<:Tropical{FT},RS}
    Tropical(VectorizationBase._vbroadcast(StaticInt{W}(), zero(T).n, StaticInt{RS}()))
end

# `gep` is a shorthand for "get element pointer"
@inline function VectorizationBase._gep(ptr::Ptr{Tropical{T}}, ::StaticInt{N}, ::StaticInt{RS}) where {N, T <: NativeTypes, RS}
    Ptr{Tropical{T}}(VectorizationBase._gep(Ptr{T}(ptr), StaticInt{N}(), StaticInt{RS}()))
end
@inline function VectorizationBase._gep(ptr::Ptr{Tropical{T}}, i::I, ::StaticInt{RS}) where {I <: IntegerIndex, T <: NativeTypes, RS}
    Ptr{Tropical{T}}(VectorizationBase._gep(Ptr{T}(ptr), i, StaticInt{RS}()))
end
@inline function VectorizationBase._gep(ptr::Ptr{Tropical{T}}, i::LazyMulAdd{M,O,I}, ::StaticInt{RS}) where {T <: NativeTypes, I <: Integer, O, M, RS}
    Ptr{Tropical{T}}(VectorizationBase._gep(Ptr{T}(ptr), i, StaticInt{RS}()))
end

for TP in [:NativeTypes, :AbstractSIMD]
    @eval @inline function Base.fma(x::Tropical{V}, y::Tropical{V}, z::Tropical{V}) where {V<:$TP}
        Tropical(Base.FastMath.max_fast(content(z), Base.FastMath.add_fast(content(x), content(y))))
    end
    @eval @inline function Base.fma(::StaticInt{N}, y::Tropical{V}, z::Tropical{V}) where {N,V<:$TP}
        Base.FastMath.add_fast(Base.FastMath.mul_fast(StaticInt{N}(), y), z)
    end

    for f ∈ [:(Base.:(*)), :(Base.FastMath.mul_fast)]
        @eval begin
            @inline $f(::StaticInt{0}, vx::Tropical{T}) where {T<:$TP} = zero(Tropical{T})
            @inline $f(::StaticInt{1}, vx::Tropical{T}) where {T<:$TP} = vx
            @inline $f(vx::Tropical{T}, ::StaticInt{0}) where {T<:$TP} = zero(Tropical{T})
            @inline $f(vx::Tropical{T}, ::StaticInt{1}) where {T<:$TP} = vx
        end
    end
    for f ∈ [:(Base.:(+)), :(Base.FastMath.add_fast)]
        @eval begin
            @inline $f(::StaticInt{0}, vx::Tropical{T}) where {T<:$TP} = vx
            @inline $f(vx::Tropical{T}, ::StaticInt{0}) where {T<:$TP} = vx
        end
    end
end
# julia 1.5 patch
@inline function VectorizationBase.VecUnroll(data::Tuple{T,Vararg{T,N}}) where {N,T<:Tropical}
    Tropical(VecUnroll(map(content, data)))
end

@inline LoopVectorization.vecmemaybe(x::Tropical) = x
@inline function VectorizationBase.collapse_add(vu::Tropical{VecUnroll{N,W,T,V}}) where {N,W,T,V}
    Tropical(VectorizationBase.collapse_max(content(vu)))
end
@inline function VectorizationBase.contract_add(vu::Tropical{VecUnroll{N,W,T,V}}, ::StaticInt{K}) where {N,W,T,V,K}
    Tropical(VectorizationBase.contract_max(content(vu), StaticInt{K}()))
end
@inline function VectorizationBase.reduced_add(x::Tropical, y::Tropical)
    Tropical(VectorizationBase.reduced_max(content(x), content(y)))
end

@inline function VectorizationBase.ifelse(f::F, m::AbstractMask, v1::Tropical, v2::Tropical, v3::Tropical) where {F}
    Tropical(VectorizationBase.ifelse(m, content(f(v1, v2, v3)), content(v3)))
end

@inline VectorizationBase.vsum(x::Tropical{<:AbstractSIMD}) = Tropical(VectorizationBase.vmaximum(content(x)))

# Overwrite the `mul!` in LinearAlgebra (also changes the behavior of `*` in Base)!
using Octavian
const XTranspose{T} = Transpose{T, <:AbstractVecOrMat{T}}
for TA in [:AbstractMatrix, :XTranspose]
    for TB in [:AbstractMatrix, :XTranspose]
        @eval function LinearAlgebra.mul!(o::AbstractMatrix{T}, a::$TA{T}, b::$TB{T}, α::Number, β::Number) where {T<:Tropical{<:NativeTypes}}
            α = _convert_to_static(T, α)
            β = _convert_to_static(T, β)
            Octavian.matmul!(o, a, b, α, β)
        end
    end
end
# NOTE: benchmark shows, the type instability here can be optimized by the compiler
# so you do not need to worry about the overheads.
@inline _convert_to_static(::Type{T}, α::TropicalTypes) where T<:TropicalTypes = α
@inline function _convert_to_static(::Type{T}, α::Number) where T<:TropicalTypes
    if iszero(α)
        return StaticInt{0}()
    elseif isone(α)
        return StaticInt{1}()
    else
        throw(ArgumentError("$α is not a valid tropical number."))
    end
end

using Octavian: zstridedpointer, preserve_buffer, matmul_sizes, matmul_params, dontpack, matmul_st_pack_dispatcher!,
    loopmul!, inlineloopmul!, maybeinline, matmul_only_β!, One, Zero, ArrayInterface, block_sizes, __matmul!

# a patch to allow tropical types
@inline function Octavian._matmul!(C::AbstractMatrix{T}, A, B, α, β, nthread, MKN) where {T<:Tropical{<:NativeTypes}}
    M, K, N = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    if M * N == 0
        return
    elseif K == 0
        matmul_only_β!(C, β)
        return
    end
    W = pick_vector_width(T)
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    mᵣ, nᵣ = matmul_params(Val(T))
    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, ArrayInterface.is_column_major(A)) # check MUST be compile-time resolvable
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        else
            (nᵣ ≥ N) && @goto LOOPMUL
            if (Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)
                (M*K*N < (StaticInt{4_096}() * W)) && @goto LOOPMUL
            else
                (M*K*N < (StaticInt{32_000}() * W)) && @goto LOOPMUL
            end
            __matmul!(pC, pA, pB, α, β, M, K, N, nthread)
            return
            @label LOOPMUL
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end

@inline function Octavian._matmul_serial!(
    C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, β, MKN
) where {T<:Tropical}
    M, K, N = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    if M * N == 0
        return
    elseif K == 0
        matmul_only_β!(C, β)
        return
    end
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    Mc, Kc, Nc = block_sizes(Val(T)); mᵣ, nᵣ = matmul_params(Val(T));
    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, ArrayInterface.is_column_major(A)) # check MUST be compile-time resolvable
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        elseif (nᵣ ≥ N) || dontpack(pA, M, K, Mc, Kc, T)
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        else
            matmul_st_pack_dispatcher!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end # function

function Octavian._matmul!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN, contig_axis) where {T<:Tropical}
  @tturbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end
function Octavian._matmul_serial!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN) where {T<:Tropical}
  @turbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end

Octavian.matmul_params(::Val{T}) where {T <: Tropical} = LoopVectorization.matmul_params()
@inline Octavian.incrementp(A::AbstractStridedPointer{<:Tropical,3}, a::Ptr) = VectorizationBase.increment_ptr(A, a, (Zero(), Zero(), One()))
@inline Octavian.increment2(B::AbstractStridedPointer{<:Tropical,2}, b::Ptr, ::StaticInt{nᵣ}) where {nᵣ} = VectorizationBase.increment_ptr(B, b, (Zero(), StaticInt{nᵣ}()))
@inline Octavian.increment1(C::AbstractStridedPointer{<:Tropical,2}, c::Ptr, ::StaticInt{mᵣW}) where {mᵣW} = VectorizationBase.increment_ptr(C, c, (StaticInt{mᵣW}(), Zero()))
