using Plots, DelimitedFiles

function data_theoretical(::Type{T}, ns) where T<:Union{Float64,Int64}
    # 0.0581s for 1000 x 1000, Float64
    map(n->n^3/1000^3*0.0581, ns)
end

function data_theoretical(::Type{T}, ns) where T<:Union{Float32}
    # 0.0581s for 1000 x 1000, Float64
    map(n->n^3/1000^3*0.0581/2, ns)
end

function plot_res(::Type{T}; dir=@__DIR__, max_threads=6) where T
    x = 2 .^ (1:12)
    ID1 = if T === Float64
        2
    elseif T === Float32
        1
    elseif T === Int
        3
    else
        error("not defined for type $T")
    end
    ax = plot(x, data_theoretical(T, x), label="theoretical", legendfont=8, legend=:topleft, ylabel="seconds", xlabel="n", xscale=:log10, yscale=:log10)
    data = reshape(readdlm(joinpath(dir, "benchmarks.dat")), 6, 12)
    plot!(ax, x, (data')[:,[2*ID1-1, 2*ID1]] ./ 1e9, label=["Float32-naive" "Float32-serial" "Float64-naive" "Float64-serial" "Int-naive" "Int-serial"][:,[2*ID1-1,2*ID1]])
    for nthreads in 2:max_threads
        data = reshape(readdlm(joinpath(dir, "benchmarks-$nthreads.dat")), 3, 12)
        plot!(ax, x, (data')[:,ID1] .* nthreads ./ 1e9, label=["Float32-$nthreads" "Float64-$nthreads" "Int-$nthreads"][ID1])
    end
    return ax
end