### plotting

println("Plotting")

using JLD2
using Plots
pyplot()
default(titlefont = (12, "serif"), guidefont = (12, "serif"), tickfont = (10, "serif"), legendfont = (8, "serif"))
PyPlot.rc("text", usetex=true)

using LaTeXStrings

## need to reload these for data loading 
include("functions_kepler.jl")


## Load data
data_file = normpath(joinpath(@__DIR__, "../data/kepler.jld2"))
if isfile(data_file)
    println("Found file from setup, loading $data_file")
    @load data_file sol optgpsol Noptsol VLB_gp2osol gpsol extraneuralsol traj longNoptsol longoptgpsol longVLB_gp2osol
else
    include("setup_kepler.jl")
end




H(x) = norm(x[3:4])^2/2 - k/(norm(x[1:2]))
I(x) = x[1]*x[4] - x[3]*x[2]

function firstintegralpots(p, sol, colors, label, ds = 1; kwargs...)
    hamil = H.(sol.u)
    plot!(p, sol.t[1:ds:end], (hamil[1:ds:end] .- H(sol.u[1]))./H(sol.u[1]) , color = colors[1], subplot = 1,
        label = label, title = "Hamiltonian", ylabel = LaTeXString("\$ (H-H(0))/H(0) \$"),
        legend = :topright, ; kwargs...)
    angmom = I.(sol.u)
    plot!(p, sol.t[1:ds:end], (angmom[1:ds:end] .- I(sol.u[1]) ) ./ I(sol.u[1]) , color = colors[1], subplot = 2,
        label = "", title = "Angular Momentum", ylabel = LaTeXString("\$ (J-J(0))/J(0) \$"); kwargs...)
end



# trajectories
pss = plot(layout = (2,2), bottom_margin = ([0 -0 0 -0].*Plots.Measures.mm), left_margin = ([0 0 0 0].*Plots.Measures.mm), 
    legend = :none,
    size = (800, 480))
idxmap = [3, 4, 1, 2]
lgndplt = 2
lwdh = 1.5
for i in 1:4
    if i == lgndplt
        lb1 = "truth"
        lb2 = "equiv. GP"
        lb3 = "low prec."
        lb4 = "2nd order"
        lb5 = "sqexp. GP"
        lb6 = "data"
        lb7 = "NN"
    else
        lb1 = lb2 = lb3 = lb4 = lb5 = lb6 = lb7 = ""
    end
    plot!(pss, sol, vars = i, linewidth = 1.3, subplot = i, label = lb1, color = :black,
        ylabel = LaTeXString("\$ x_$i \$"))
    plot!(pss, optgpsol, vars = i, linewidth = lwdh, subplot = i, label = lb2, 
        color = Plots.RGB(([170, 110, 40]./255)...))
    plot!(pss, Noptsol, vars = i, subplot = i, label = lb3, 
        color = Plots.RGB(([255, 225, 25]./255)...), linestyle = :solid, linewidth = lwdh)
    plot!(pss, VLB_gp2osol, vars = idxmap[i], linewidth = lwdh, subplot = i, label = lb4, 
        color = Plots.RGB(([0, 130, 200]./255)...))
    plot!(pss, gpsol, vars = i, linewidth = lwdh, subplot = i, label = lb5, 
        color = Plots.RGB(([220, 190, 255]./255)...))
    plot!(pss, extraneuralsol, vars = i, subplot = i, label = lb7, 
        color = Plots.RGB(([128, 0, 0]./255)...),
        linewidth = lwdh)
    
    scatter!(pss, traj.t, getindex.(traj.u, i), subplot = i, label = lb6, color = :black, markersize = 1.6)
end



# long term 
ds = 60
lw = 1.7
p = plot(size = (400, 400), layout = (2,1))
# firstintegralpots(p, sol,   [:black], "truth")
plot!(p, [0,longNoptsol.t[end]], [0,0], color = :black, subplot = 1)
plot!(p, [0,longNoptsol.t[end]], [0,0], color = :black, subplot = 2, label = "")
firstintegralpots(p, longNoptsol, [Plots.RGB(([255, 225, 25]./255)...)], "low prec.", ds; 
    linewidth = lw)
firstintegralpots(p, longoptgpsol, [Plots.RGB(([170, 110, 40]./255)...)], "equiv. GP", ds; 
    linewidth = lw)
firstintegralpots(p, longVLB_gp2osol, [Plots.RGB(([0, 130, 200]./255)...)], "2nd order", ds; 
    linewidth = lw, guidefont = (10, "serif"))
plot!(p, subplot = 1, label = "", legend = :none)


sdoffst = 0
offset2 = 10
bothplots = plot(pss, p, size = (800, 330), 
left_margin = ([0 -sdoffst 0 -sdoffst offset2 offset2].*Plots.Measures.mm), right_margin = ([0 sdoffst 0 sdoffst 0 0].*Plots.Measures.mm))

plot_file = normpath(joinpath(@__DIR__, "../plots/kepler.png"))
savefig(bothplots, plot_file)