using TropicalGEMM, BenchmarkTools
using Octavian
using DelimitedFiles

function run_benchmarks(cases, suite; output_file)
    tune!(suite)
    res = run(suite)

    times = zeros(length(cases))
    for (k, case) in enumerate(cases)
        times[k] = minimum(res[case].times)
    end

    println("Writing benchmark results to file: $output_file.")
    mkpath(dirname(output_file))
    writedlm(output_file, times)
end

function naivemm!(o::Matrix, a::Matrix, b::Matrix)
    @assert size(a, 2) == size(b, 1) && size(o) == (size(a, 1), size(b, 2))
    for j=1:size(b, 2)
        for k=1:size(a, 2)
            for i=1:size(a, 1)
                @inbounds o[i,j] += a[i,k] * b[k,j]
            end
        end
    end
    return o
end

function generate_suite(; multithreading)
    cases = []
    suite = BenchmarkGroup()
    for s=1:12
        for T in (Float32, Float64, Int)
            n = 1<<s
            if multithreading
                push!(cases, "octavian-$T-$n")
                suite["octavian-$T-$n"] = @benchmarkable Octavian.matmul!(o, a, b) setup=(a = Tropical.(ones($T, $n, $n)); b = Tropical.(ones($T, $n, $n)); o = zeros(Tropical{$T}, $n, $n))
            else
                push!(cases, "naive-$T-$n")
                suite["naive-$T-$n"] = @benchmarkable naivemm!(o, a, b) setup=(a = Tropical.(ones($T, $n, $n)); b = Tropical.(ones($T, $n, $n)); o = zeros(Tropical{$T}, $n, $n))
                push!(cases, "octavian-$T-$n")
                suite["octavian-$T-$n"] = @benchmarkable Octavian.matmul_serial!(o, a, b) setup=(a = Tropical.(ones($T, $n, $n)); b = Tropical.(ones($T, $n, $n)); o = zeros(Tropical{$T}, $n, $n))
            end
        end
    end
    cases, suite
end

nthreads = Base.Threads.nthreads()
if  nthreads == 1
    cases, suite = generate_suite(multithreading=false)
    run_benchmarks(cases, suite; output_file=joinpath(@__DIR__, "benchmarks.dat"))
else
    cases, suite = generate_suite(multithreading=true)
    run_benchmarks(cases, suite; output_file=joinpath(@__DIR__, "benchmarks-$nthreads.dat"))
end