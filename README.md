# TropicalGEMM

[![Build Status](https://github.com/TensorBFS/TropicalGEMM.jl/workflows/CI/badge.svg)](https://github.com/TensorBFS/TropicalGEMM.jl/actions)

STILL WORK IN PROGRESS!

## See the discussion here

https://github.com/JuliaSIMD/LoopVectorization.jl/issues/201

### Get started

Open a Julia REPL and type `]` to enter the `pkg>` mode, and then install related packages with
```julia
pkg> add TropicalNumbers, Octavian, TropicalGEMM, BenchmarkTools
```

In a julia REPL, you can try a minimum working example
```julia
julia> using TropicalNumbers, Octavian, TropicalGEMM, BenchmarkTools

julia> a = Tropical.(randn(1000, 1000))

julia> @benchmark Octavian.matmul_serial($a, $a)
```