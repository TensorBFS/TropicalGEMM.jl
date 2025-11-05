const BlasSemiringTypes{T} = Union{Tropical{T}, TropicalMinPlus{T}, TropicalMaxMul{T}, TropicalMaxMin{T}, TropicalBitwise{T}}
basetype(::Type{<:Tropical}) = Tropical
basetype(::Type{<:TropicalMinPlus}) = TropicalMinPlus
basetype(::Type{<:TropicalMaxMul}) = TropicalMaxMul
basetype(::Type{<:TropicalMaxMin}) = TropicalMaxMin
basetype(::Type{<:TropicalBitwise}) = TropicalBitwise

# implement neginf for Vec and VecUnroll
TropicalNumbers.neginf(::Type{Vec{N, T}}) where {N,T} = Vec(ntuple(i->neginf(T), N)...)
TropicalNumbers.neginf(::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V} = VecUnroll(ntuple(i->neginf(V), N+1))

LoopVectorization.check_args(::Type{T}, ::Type{T}) where T<:BlasSemiringTypes = true
LoopVectorization.check_type(::Type{<:BlasSemiringTypes{T}}) where {T} = LoopVectorization.check_type(T)

for TT in [:Tropical, :TropicalMinPlus, :TropicalMaxMul, :TropicalMaxMin, :TropicalBitwise]
    @eval @inline function VectorizationBase._vstore!(
        ptr::AbstractStridedPointer, vu::$TT{<:VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
        VectorizationBase._vstore!(notropical(ptr), content(vu), u, a, s, nt, si)
    end

    @eval @inline function VectorizationBase._vstore!(ptr::AbstractStridedPointer, vu::$TT{<:VecUnroll{Nm1,W}}, u::Unroll{AU,F,N,AV,W}, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
        VectorizationBase._vstore!(notropical(ptr), content(vu), u, m, a, s, nt, si)
    end

    @eval @inline function VectorizationBase._vstore!(
        g::G, ptr::AbstractStridedPointer{T,D,C}, vu::$TT{<:VecUnroll{U,W}}, u::Unroll{AU,F,N,AV,1,M,X,I}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X}
        VectorizationBase._vstore!(g, notropical(ptr), content(vu), u, a, s, nt, si)
    end

    @eval @inline function VectorizationBase.__vstore!(
        f::F, ptr::Ptr{$TT{T}}, v::$TT{T}, i::IntegerIndex, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {T<:NativeTypesExceptBit, F<:Function,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
        VectorizationBase.__vstore!(f, Ptr{T}(ptr), content(v), i, a, s, nt, si)
    end

    @eval @inline function VectorizationBase.__vstore!(
        ptr::Ptr{$TT{T}}, v::$TT{VT}, i::I, m::AbstractMask{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {W, T <: NativeTypesExceptBit, VT <: Vec, I <: Index, A <: StaticBool, S <: StaticBool, NT <: StaticBool, RS}
        VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, m, a, s, nt, si)
    end
    @eval @inline function VectorizationBase.__vstore!(
        ptr::Ptr{$TT{T}}, v::$TT{VT}, i::VectorIndex{W}, a::A, s::S, nt::NT, si::StaticInt{RS}
    ) where {T,VT<:Vec,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS}
        VectorizationBase.__vstore!(Ptr{T}(ptr), content(v), i, a, s, nt, si)
    end
    @eval @inline function VectorizationBase.__vload(ptr::Ptr{$TT{T}}, i::I, m::AbstractMask, a::A, si::StaticInt{RS})  where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
        $TT(VectorizationBase.__vload(Ptr{T}(ptr), i, m, a, si))
    end
    @eval @inline function VectorizationBase.__vload(ptr::Ptr{$TT{T}}, i::I, a::A, si::StaticInt{RS}) where {A <: StaticBool, T <: NativeTypes, I <: Index, RS}
        $TT(VectorizationBase.__vload(Ptr{T}(ptr), i, a, si))
    end
    @eval @inline function VectorizationBase._vbroadcast(a::Union{Val{W},StaticInt{W}}, s::$TT{T}, si::StaticInt{RS}) where {W,T,RS}
        $TT(VectorizationBase._vbroadcast(a, content(s), si))
    end
    @eval @inline function VectorizationBase._vload(ptr::AbstractStridedPointer{$TT{T}}, u::Unroll, ::A, ::StaticInt{RS}) where {T,A<:StaticBool,RS}
        res = VectorizationBase._vload(notropical(ptr), u, A(), StaticInt{RS}())
        $TT(res)
    end
    @eval @inline function VectorizationBase._vload(ptr::AbstractStridedPointer{$TT{T}}, u::Unroll, m::AbstractMask, ::A, ::StaticInt{RS}) where {T,A<:StaticBool,RS}
        res = VectorizationBase._vload(notropical(ptr), u, m, A(), StaticInt{RS}())
        $TT(res)
    end
    @eval @inline function Base.promote_rule(::Type{$TT{T1}}, ::Type{$TT{T2}}) where {T1<:VecUnroll,T2<:Vec}
        $TT{promote_rule(T1, T2)}
    end
end

@inline function notropical(ptr::VectorizationBase.StridedPointer{<:BlasSemiringTypes{T},N,C,B,R,X,O}) where {T,N,C,B,R,X,O}
    stridedpointer(Ptr{T}(ptr.p), ptr.si, StaticInt{B}())
end

@inline function notropical(ptr::OffsetPrecalc{<:BlasSemiringTypes})
    VectorizationBase.OffsetPrecalc(notropical(ptr.ptr), ptr.precalc)
end

@generated function VectorizationBase.zero_vecunroll(::StaticInt{N}, ::StaticInt{W}, ::Type{TT}, ::StaticInt{RS}) where {N,W,T,RS,TT<:BlasSemiringTypes{T}}
    quote
        $(Expr(:meta,:inline))
        t = Base.Cartesian.@ntuple $N n -> VectorizationBase._vbroadcast(StaticInt{$W}(), $(zero(TT).n), StaticInt{$RS}())
        $(basetype(TT))(VecUnroll(t))
    end
end

@inline function VectorizationBase._vzero(::StaticInt{W}, ::Type{TT}, ::StaticInt{RS}) where {W,FT,TT<:BlasSemiringTypes{FT},RS}
    basetype(TT)(VectorizationBase._vbroadcast(StaticInt{W}(), zero(TT).n, StaticInt{RS}()))
end

# `gep` is a shorthand for "get element pointer"
@inline function VectorizationBase._gep(ptr::Ptr{TT}, ::StaticInt{N}, ::StaticInt{RS}) where {N, T <: NativeTypes, TT<:BlasSemiringTypes{T}, RS}
    Ptr{TT}(VectorizationBase._gep(Ptr{T}(ptr), StaticInt{N}(), StaticInt{RS}()))
end
@inline function VectorizationBase._gep(ptr::Ptr{TT}, i::I, ::StaticInt{RS}) where {I <: IntegerIndex, T <: NativeTypes, TT<:BlasSemiringTypes{T}, RS}
    Ptr{TT}(VectorizationBase._gep(Ptr{T}(ptr), i, StaticInt{RS}()))
end
@inline function VectorizationBase._gep(ptr::Ptr{TT}, i::LazyMulAdd{M,O,I}, ::StaticInt{RS}) where {T <: NativeTypes, TT<:BlasSemiringTypes{T}, I <: Integer, O, M, RS}
    Ptr{TT}(VectorizationBase._gep(Ptr{T}(ptr), i, StaticInt{RS}()))
end

for TP in [:NativeTypes, :AbstractSIMD]
    for (TT, F0, F1) in [
        (:Tropical, :max_fast, :add_fast),
        (:TropicalMinPlus, :min_fast, :add_fast),
        (:TropicalMaxMul, :max_fast, :mul_fast),
        (:TropicalMaxMin, :max_fast, :min_fast),
    ]
        @eval @inline function Base.fma(x::$TT{V}, y::$TT{V}, z::$TT{V}) where {V<:$TP}
            $TT(Base.FastMath.$F0(content(z), Base.FastMath.$F1(content(x), content(y))))
        end
    end
    @eval @inline function Base.fma(::StaticInt{N}, y::TT, z::TT) where {N,T<:$TP,TT<:BlasSemiringTypes{T}}
        Base.FastMath.add_fast(Base.FastMath.mul_fast(StaticInt{N}(), y), z)
    end

    for f ∈ [:(Base.:(*)), :(Base.FastMath.mul_fast)]
        @eval begin
            @inline $f(::StaticInt{0}, vx::TT) where {T<:$TP, TT<:BlasSemiringTypes{T}} = zero(TT)
            @inline $f(::StaticInt{1}, vx::BlasSemiringTypes{T}) where {T<:$TP} =  vx
            @inline $f(vx::TT, ::StaticInt{0}) where {T<:$TP, TT<:BlasSemiringTypes{T}} = zero(TT)
            @inline $f(vx::BlasSemiringTypes{T}, ::StaticInt{1}) where {T<:$TP} = vx
        end
    end
    for f ∈ [:(Base.:(+)), :(Base.FastMath.add_fast)]
        @eval begin
            @inline $f(::StaticInt{0}, vx::BlasSemiringTypes{T}) where {T<:$TP} = vx
            @inline $f(vx::BlasSemiringTypes{T}, ::StaticInt{0}) where {T<:$TP} = vx
        end
    end
end
# julia 1.5 patch
@inline function VectorizationBase.VecUnroll(data::Tuple{T,Vararg{T,N}}) where {N,T<:BlasSemiringTypes}
    basetype(T)(VecUnroll(map(content, data)))
end

@inline LoopVectorization.vecmemaybe(x::BlasSemiringTypes) = x

for (TT, F0, F1, F2, F3, F4) in [
        (:Tropical, :max_fast, :collapse_max, :contract_max, :reduced_max, :vmaximum),
        (:TropicalMinPlus, :min_fast, :collapse_min, :contract_min, :reduced_min, :vminimum),
        (:TropicalMaxMul, :max_fast, :collapse_max, :contract_max, :reduced_max, :vmaximum),
        (:TropicalMaxMin, :max_fast, :collapse_max, :contract_max, :reduced_max, :vmaximum),
        (:TropicalBitwise, :|,       :collape_or,   :contract_or,  :reduced_any, :vany),
    ]
    @eval @inline Base.FastMath.add_fast(a::$TT, b::$TT) = $TT(Base.FastMath.$F0(content(a), content(b)))

    @eval @inline function VectorizationBase.collapse_add(vu::$TT{VecUnroll{N,W,T,V}}) where {N,W,T,V}
        $TT(VectorizationBase.$F1(content(vu)))
    end
    @eval @inline function VectorizationBase.contract_add(vu::$TT{VecUnroll{N,W,T,V}}, ::StaticInt{K}) where {N,W,T,V,K}
        $TT(VectorizationBase.$F2(content(vu), StaticInt{K}()))
    end
    @eval @inline function VectorizationBase.reduced_add(x::$TT, y::$TT)
        $TT(VectorizationBase.$F3(content(x), content(y)))
    end
    @eval @inline VectorizationBase.vsum(x::$TT{<:AbstractSIMD}) = $TT(VectorizationBase.$F4(content(x)))
    @eval @inline function VectorizationBase.ifelse(f::F, m::AbstractMask, v1::$TT, v2::$TT, v3::$TT) where {F<:Function}
        $TT(VectorizationBase.ifelse(m, content(f(v1, v2, v3)), content(v3)))
    end
    @eval @inline function VectorizationBase.vifelse(f::F, m::AbstractMask, a::$TT, b::$TT, c::$TT) where {F<:Function}
        VectorizationBase.vifelse(m, f(a, b, c), c)
    end
    @eval @inline function VectorizationBase.vifelse(m::AbstractMask, a::$TT, b::$TT)
        $TT(VectorizationBase.vifelse(m, content(a), content(b)))
    end
end

# Overwrite the `mul!` in LinearAlgebra (also changes the behavior of `*` in Base)!
using Octavian
function LinearAlgebra.mul!(o::MaybeAdjOrTransMat{T}, a::MaybeAdjOrTransMat{T}, b::MaybeAdjOrTransMat{T}, α::Number, β::Number) where {T<:BlasSemiringTypes{<:NativeTypes}}
    α = _convert_to_static(T, α)
    β = _convert_to_static(T, β)
    Octavian.matmul!(o, a, b, α, β)
end
# NOTE: benchmark shows, the type instability here can be optimized by the compiler
# so you do not need to worry about the overheads.
@inline _convert_to_static(::Type{T}, α::AbstractSemiring) where T<:AbstractSemiring = α
@inline function _convert_to_static(::Type{T}, α::Number) where T<:AbstractSemiring
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
@inline function Octavian._matmul!(C::AbstractMatrix{T}, A, B, α, β, nthread, MKN) where {T<:BlasSemiringTypes{<:NativeTypes}}
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
) where {T<:BlasSemiringTypes}
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

function Octavian._matmul!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN, contig_axis) where {T<:BlasSemiringTypes}
  @tturbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end
function Octavian._matmul_serial!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN) where {T<:BlasSemiringTypes}
  @turbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end

Octavian.matmul_params(::Val{T}) where {T <: BlasSemiringTypes} = LoopVectorization.matmul_params()
@inline Octavian.incrementp(A::AbstractStridedPointer{<:BlasSemiringTypes,3}, a::Ptr) = VectorizationBase.increment_ptr(A, a, (Zero(), Zero(), One()))
@inline Octavian.increment2(B::AbstractStridedPointer{<:BlasSemiringTypes,2}, b::Ptr, ::StaticInt{nᵣ}) where {nᵣ} = VectorizationBase.increment_ptr(B, b, (Zero(), StaticInt{nᵣ}()))
@inline Octavian.increment1(C::AbstractStridedPointer{<:BlasSemiringTypes,2}, c::Ptr, ::StaticInt{mᵣW}) where {mᵣW} = VectorizationBase.increment_ptr(C, c, (StaticInt{mᵣW}(), Zero()))
