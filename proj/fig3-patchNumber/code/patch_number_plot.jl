using Plots
using Colors
using Images
using DifferentialEquations
using Statistics
using JLD2
using DataFrames
using CSV

# For each random seeds
# For each volume used
# Record number of patches

seeds = (1,2,3,4,5,6,7,8,9,10,11,12)
volumes = (50,100,200,300,400,500,600,700,800,900)

avg_n = []
std_n = []
max_n = []
min_n = []

for volume in volumes
    n = []
    for seed in seeds
        sol = load("../../Simulations/sims/simple_30min_seed_$(seed)_volume_$(volume).jld2", "sol_simp");
        locs = findlocalmaxima(sol[:,:,1,end], edges=false)
        maxima = sol[:,:,1,end][locs]
        maxima = maxima[maxima .> 10]
        append!(n, length(maxima))
    end
    append!(avg_n, mean(n))
    append!(min_n, minimum(n))
    append!(max_n, maximum(n))
    append!(std_n, std(n)*(!isnan(std(n)))) # workaround to transform NaN to 0
end

experimental = CSV.read("../data/siteNumber.csv", DataFrame)
exp_rad = experimental[!, "EffectiveRadius"]
exp_n = experimental[!, "Bud Number"]
exp_v = experimental[!, "NA"].* ((4/3)*(0.5^3)*pi);

volumes = [50,100,200,300,400,500,600,700,800,900]
plot(volumes, avg_n, linewidth=3, ribbon=std_n, fillalpha=.5, label="Simulated", xaxis="Volume", yaxis="Patch number", color=1)
plot!(volumes, avg_n, ribbon=(max_n-avg_n, avg_n-min_n), fillalpha=.2, label=false, xaxis="Volume", yaxis="Patch number", color=1)
scatter!(exp_v, exp_n, color="red", label="Measured Cells", alpha=0.7)
savefig("../figures/N_by_V_tmp.png")