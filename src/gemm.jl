using TropicalNumbers, VectorizationBase
using LoopVectorization
using VectorizationBase: OffsetPrecalc, StaticBool, Bit, static, NativeTypes, Index, gep_quote, VectorIndex,
    AbstractMask, NativeTypesExceptBit, AbstractSIMDVector, IndexNoUnroll, AbstractStridedPointer
using VectorizationBase: contiguous_batch_size, contiguous_axis, val_stride_rank, bytestrides, offsets, memory_reference

LoopVectorization.check_args(::Type{T}, ::Type{T}) where T<:Tropical = true
LoopVectorization.check_type(::Type{Tropical{T}}) where {T} = LoopVectorization.check_type(T)

@inline VectorizationBase.vstore!(ptr::VectorizationBase.StridedPointer{T}, v::T) where {T<:Tropical} = vstore!(ptr, content(v))
@inline function VectorizationBase.vstore!(
    ptr::Ptr{Tropical{T}}, v::Tropical{Vec{N,T}}, i::VectorIndex{W}, m::VectorizationBase.AbstractSIMDVector{W}, a::A, s::S, nt::NT, si::StaticInt{RS}) where {T,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,N}
    vstore!(convert(Ptr{T}, ptr), content(v), i, m, a, s, nt, si)
end
@inline function VectorizationBase.vstore!(
    ptr::Ptr{Tropical{T}}, v::Tropical{Vec{N,T}}, m::VectorizationBase.AbstractSIMDVector{W}, a::A, s::S, nt::NT, si::StaticInt{RS}) where {T,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,N}
    vstore!(convert(Ptr{T}, ptr), content(v), m, a, s, nt, si)
end

@inline function VectorizationBase._vstore!(
    ptr::AbstractStridedPointer, vu::Tropical{<:VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, a::A, s::S, nt::NT, si::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
    VectorizationBase._vstore!(notropical(ptr), content(vu), u, a, s, nt, si)
end
@inline function VectorizationBase.__vstore!(
        ptr::Ptr{Tropical{T}}, v::Tropical{VT}, i::I, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {W, T <: NativeTypesExceptBit, VT <: NativeTypes, I <: Index, A <: StaticBool, S <: StaticBool, NT <: StaticBool, RS}
    VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, m, a, s, nt, si)
end

@inline function VectorizationBase.__vstore!(
        ptr::Ptr{Tropical{T}}, v::Tropical{VT}, i::I, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {W, T <: NativeTypesExceptBit, VT <: Vec, I <: Index, A <: StaticBool, S <: StaticBool, NT <: StaticBool, RS}
    VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, m, a, s, nt, si)
end

#@inline function VectorizationBase._vstore!(
#    ptr::OffsetPrecalc, vu::Tropical{VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, ::A, ::S, ::NT, ::StaticInt{RS}
#) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
#    VectorizationBase._vstore!(notropical(ptr), content(vu), u, a, s, nt, si)
#end

@inline function VectorizationBase.vload(ptr::Ptr{Tropical{T}}, i::I, m::Mask, a::A, si::StaticInt{RS}) where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
    Tropical(vload(Ptr{T}(ptr), i, m, a, si))
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

@inline function VectorizationBase._vload(ptr::AbstractStridedPointer{Tropical{T}}, u::Unroll, a::A, si::StaticInt{RS}) where {T,A<:StaticBool,RS}
    res = VectorizationBase._vload(notropical(ptr), u, a, si)
    Tropical(res)
end

@inline function VectorizationBase.zero_vecunroll(n::StaticInt{N}, w::StaticInt{W}, ::Type{Tropical{T}}, si::StaticInt{RS}) where {N,W,T,RS}
    res = Tropical(VectorizationBase.zero_vecunroll(n, w, T, si))
    return res
end

@inline function Base.promote_rule(::Type{Tropical{T1}}, ::Type{Tropical{T2}}) where {T1<:VecUnroll,T2<:Vec}
    Tropical{promote_rule(T1, T2)}
end

@inline function VectorizationBase._vzero(::StaticInt{W}, ::Type{T}, ::StaticInt{RS}) where {W,T<:Tropical{FT},RS} where FT
    Tropical(VectorizationBase._vbroadcast(StaticInt{W}(), FT(-Inf), StaticInt{RS}()))
end

@inline function VectorizationBase.fma(x::Tropical{V}, y::Tropical{V}, z::Tropical{V}) where {V<:VectorizationBase.AbstractSIMD}
    Tropical(max(content(z), content(x) + content(y)))
end

@inline function VectorizationBase.similar_no_offset(sptr::OffsetPrecalc{T}, ptr::Ptr{Tropical{T}}) where {T}
    OffsetPrecalc(VectorizationBase.similar_no_offset(getfield(sptr, :ptr), ptr), getfield(sptr, :precalc))
end

# is `gep` a shorthand for "get element pointer"?
@inline VectorizationBase.gep(ptr::Ptr{Tropical{T}}, i) where T = Ptr{Tropical{T}}(VectorizationBase.gep(Ptr{T}(ptr), i))

# TODO: FIX!!!!!!
@inline function Base.promote(a::Int, b::Tropical{T}) where {T<:Vec}
    elem = a == 0 ? -Inf : 0.0
    Tropical(T(elem)), b
end

@inline function Base.promote(a::Int, b::Tropical{T}, c::Tropical{T}) where {T<:Vec}
    elem = a == 0 ? -Inf : 0.0
    Tropical(T(elem)), b, c
end

@inline function Base.promote(a::Int, b::Tropical{T}) where {T<:VecUnroll}
    elem = a == 0 ? -Inf : 0.0
    Tropical(T(elem)), b
end

@inline function Base.promote(a::Int, b::Tropical{T}, c::Tropical{T}) where {T<:VecUnroll}
    elem = a == 0 ? -Inf : 0.0
    Tropical(T(elem)), b, c
end

# julia 1.5 patch
@inline function VectorizationBase.VecUnroll(data::Tuple{T,Vararg{T,N}}) where {N,T<:Tropical}
    Tropical.(VecUnroll(content.(data)))
end
