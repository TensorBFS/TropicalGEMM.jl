using TropicalNumbers, VectorizationBase
using LoopVectorization
using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer, AbstractSIMD
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference

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

@inline function VectorizationBase.__vstore!(
        ptr::Ptr{Tropical{T}}, v::Tropical{VT}, i::I, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {W, T <: NativeTypesExceptBit, VT <: Vec, I <: Index, A <: StaticBool, S <: StaticBool, NT <: StaticBool, RS}
    VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, m, a, s, nt, si)
end

@inline function VectorizationBase.__vload(ptr::Ptr{Tropical{T}}, i::I, m::AbstractMask, a::A, si::StaticInt{RS})  where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
    Tropical(VectorizationBase.__vload(Ptr{T}(ptr), i, m, a, si))
end
@inline function VectorizationBase.__vload(ptr::Ptr{Tropical{T}}, i::I, a::A, si::StaticInt{RS}) where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
    Tropical(VectorizationBase.__vload(Ptr{T}(ptr), i, a, si))
end

@inline function VectorizationBase.vbroadcast(a::Union{Val{W},StaticInt{W}}, s::Tropical{T}) where {W,T}
    Tropical(VectorizationBase.vbroadcast(a, content(s)))
end

@inline function VectorizationBase.stridedpointer(A::AbstractArray{T}) where {T <: Tropical}
    p, r = memory_reference(A)
    stridedpointer(p, contiguous_axis(A), contiguous_batch_size(A), val_stride_rank(A), bytestrides(A), offsets(A))
end

@inline function VectorizationBase.stridedpointer(
    ptr::Ptr{T}, ::StaticInt{C}, ::StaticInt{B}, ::Val{R}, strd::X, offsets::O
) where {T<:Tropical,C,B,R,N,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
    VectorizationBase.StridedPointer{T,N,C,B,R,X,O}(ptr, strd, offsets)
end

@inline function notropical(ptr::VectorizationBase.StridedPointer{Tropical{T},N,C,B,R,X,O}) where {T,N,C,B,R,X,O}
    VectorizationBase.StridedPointer{T,N,C,B,R,X,O}(Ptr{T}(ptr.p), ptr.strd, ptr.offsets)
end

@inline function notropical(ptr::OffsetPrecalc{<:Tropical})
    VectorizationBase.OffsetPrecalc(notropical(ptr.ptr), ptr.precalc)
end

@inline function VectorizationBase._vload(ptr::AbstractStridedPointer{Tropical{T}}, u::Unroll, ::A, ::StaticInt{RS}) where {T,A<:StaticBool,RS}
    res = VectorizationBase._vload(notropical(ptr), u, A(), StaticInt{RS}())
    Tropical(res)
end

@generated function VectorizationBase.zero_vecunroll(::StaticInt{N}, ::StaticInt{W}, ::Type{Tropical{T}}, ::StaticInt{RS}) where {N,W,T,RS}
    quote
        $(Expr(:meta,:inline))
        t = Base.Cartesian.@ntuple $N n -> VectorizationBase._vbroadcast(StaticInt{$W}(), $(T(-Inf)), StaticInt{$RS}())
        Tropical(VecUnroll(t))
    end
end

@inline function Base.promote_rule(::Type{Tropical{T1}}, ::Type{Tropical{T2}}) where {T1<:VecUnroll,T2<:Vec}
    Tropical{promote_rule(T1, T2)}
end

@inline function VectorizationBase._vzero(::StaticInt{W}, ::Type{T}, ::StaticInt{RS}) where {W,FT,T<:Tropical{FT},RS}
    Tropical(VectorizationBase._vbroadcast(StaticInt{W}(), FT(-Inf), StaticInt{RS}()))
end

@inline function Base.fma(x::Tropical{V}, y::Tropical{V}, z::Tropical{V}) where {V<:VectorizationBase.AbstractSIMD}
    Tropical(Base.FastMath.max_fast(content(z), Base.FastMath.add_fast(content(x), content(y))))
end
@inline function Base.fma(::StaticInt{N}, y::Tropical{V}, z::Tropical{V}) where {N,V<:VectorizationBase.AbstractSIMD}
    Base.FastMath.add_fast(Base.FastMath.mul_fast(StaticInt{N}(), y), z)
end

# `gep` is a shorthand for "get element pointer"
@inline VectorizationBase.gep(ptr::Ptr{Tropical{T}}, i) where T = Ptr{Tropical{T}}(VectorizationBase.gep(Ptr{T}(ptr), i))

for f ∈ [:(Base.:(*)), :(Base.FastMath.mul_fast)]
    @eval begin
        @inline $f(::StaticInt{0}, vx::Tropical{T}) where {T<:AbstractSIMD} = zero(Tropical{T})
        @inline $f(::StaticInt{1}, vx::Tropical{T}) where {T<:AbstractSIMD} = vx
        @inline $f(vx::Tropical{T}, ::StaticInt{0}) where {T<:AbstractSIMD} = zero(Tropical{T})
        @inline $f(vx::Tropical{T}, ::StaticInt{1}) where {T<:AbstractSIMD} = vx
    end
end
for f ∈ [:(Base.:(+)), :(Base.FastMath.add_fast)]
    @eval begin
        @inline $f(::StaticInt{0}, vx::Tropical{T}) where {T<:AbstractSIMD} = vx
        @inline $f(vx::Tropical{T}, ::StaticInt{0}) where {T<:AbstractSIMD} = vx
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

