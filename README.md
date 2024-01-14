# TropicalGEMM

[![Build Status](https://github.com/TensorBFS/TropicalGEMM.jl/workflows/CI/badge.svg)](https://github.com/TensorBFS/TropicalGEMM.jl/actions)
[![codecov](https://codecov.io/gh/TensorBFS/TropicalGEMM.jl/branch/master/graph/badge.svg?token=8F6PH5Q9PL)](https://codecov.io/gh/TensorBFS/TropicalGEMM.jl)

The fastest Tropical matrix multiplication in the world! Supported matrix element types include
* max-plus algebra: `Tropical{BlasType}`
* min-plus algebra numbers: `TropicalMinPlus{BlasType}`
* max-times algebra numbers: `TropicalMaxMul{BlasType}`

Please check [`TropicalNumbers.jl`](https://github.com/TensorBFS/TropicalNumbers.jl) for the definitions of these types. The `BlasType` is the storage type, which could be one of `Bool, Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8, SIMDTypes.Bit`.

## Get started

Open a Julia REPL and type `]` to enter the `pkg>` mode, and then install related packages with
```julia
pkg> add TropicalGEMM, BenchmarkTools, TropicalNumbers
```
Loading `TropicalGEMM` module into the workspace affects the `*` on Tropical matrices immediately. The following is a minimum working example
```julia
julia> using TropicalNumbers, BenchmarkTools

julia> a = Tropical.(randn(1000, 1000));

julia> @btime $a * $a;
  2.588 s (6 allocations: 7.66 MiB)

julia> using TropicalGEMM

julia> @btime $a * $a;
  66.916 ms (2 allocations: 7.63 MiB)
```

## Benchmarks

Matrix size `n x n`, CPU Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz.
The benchmark and plotting scripts could be found in the benchmarks folder.

![Float64](benchmarks/benchmark-float64.png)
![Float32](benchmarks/benchmark-float32.png)

## References
1. This package originates from the following issue:
https://github.com/JuliaSIMD/LoopVectorization.jl/issues/201
2. For applications, please check the papers listed in the [CITATION.bib](/CITATION.bib).