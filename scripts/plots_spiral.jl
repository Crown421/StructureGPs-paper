using DifferentialEquations, Flux, DiffEqFlux
using GaussianProcessODEs
using JLD2
using LaTeXStrings

using Plots
pyplot()
default(titlefont = (12, "serif"), guidefont = (12, "serif"), tickfont = (10, "serif"), legendfont = (8, "serif"))
PyPlot.rc("text", usetex=true)

# need to redefine this for the load to work
function trueODEfunc!(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
function trueODEfunc(u,p,t)
    du = similar(u)
    trueODEfunc!(du, u, p, t)
end

if isfile("../data/spiral.jld2")
    println("Found file from setup, loading ...")
    @load "../data/spiral.jld2" sol hvars extraneuralsol fitc_mstd fitc_zero_mstd datatsteps ode_data Z fitc_gpODE fitc_zero_goODE LTsol LTgpOde1 LTgpOde2 LTnnODE realU fitcU fitc_zero_U npODEU nn_U Zqg Uqg ts
else
    include("setup_spiral.jl")
end


###################################################################################################
## Figure 1
###################################################################################################

punc = plot(size = (800, 470), layout = (2, 1), legend = :outertopright)
thinlwh = 0.5
sb1 = 1
i = 1
lwh = 1.8
plot!(punc, sol, vars = 1, color = :black, label = "", subplot = sb1, ylabel = LaTeXString("\$ x_1 \$"))
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i], ribbon=hvars["uncerth"][i] ,label = "",
        color = :blue3, fillalpha = .4, linewidth = lwh, subplot = sb1)
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i] .+ hvars["uncerth"][i], label = "", color = :blue3, linewidth = thinlwh, subplot = sb1)
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i] .- hvars["uncerth"][i], label = "", color = :blue3, linewidth = thinlwh, subplot = sb1)

plot!(punc, extraneuralsol, vars = 1, subplot = sb1, linewidth = lwh + 0.4, color = [:mediumseagreen], 
    linestyle = :dash, label = "")

plot!(punc, ts, fitc_mstd[1][i,:], ribbon=fitc_mstd[2][i,:], label = "",
        color = :firebrick, fillalpha=.4, linewidth = lwh, subplot = sb1)
plot!(punc, ts, fitc_mstd[1][i,:] .+ fitc_mstd[2][i,:], label = "", color = :firebrick, linewidth = thinlwh, subplot = sb1)
plot!(punc, ts, fitc_mstd[1][i,:] .- fitc_mstd[2][i,:], label = "", color = :firebrick, linewidth = thinlwh, subplot = sb1)

plot!(punc, ts, fitc_zero_mstd[1][i,:], ribbon=fitc_zero_mstd[2][i,:], label = "",
        color = :darkgoldenrod, fillalpha=.4, linewidth = lwh, subplot = sb1)
plot!(punc, ts, fitc_zero_mstd[1][i,:] .+ fitc_zero_mstd[2][i,:], label = "", color = :darkgoldenrod, linewidth = thinlwh, subplot = sb1)
plot!(punc, ts, fitc_zero_mstd[1][i,:] .- fitc_zero_mstd[2][i,:], label = "", color = :darkgoldenrod, linewidth = thinlwh, subplot = sb1)

scatter!(punc, datatsteps, ode_data[1,:], subplot = sb1, color = :black, markersize = 3.4, label = "")  

sb1 = 2
i = 2
plot!(punc, sol, vars = 2, color = :black, label = "truth", subplot = sb1, ylabel = LaTeXString("\$ x_2 \$"))
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i], ribbon=hvars["uncerth"][i] ,label = "npODE",
        color = :blue3, fillalpha = .4, linewidth = lwh, subplot = sb1)
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i] .+ hvars["uncerth"][i], label = "", color = :blue3, linewidth = thinlwh, subplot = sb1)
plot!(punc, hvars["tsh"][:], hvars["tpredh"][:,i] .- hvars["uncerth"][i], label = "", color = :blue3, linewidth = thinlwh, subplot = sb1)

plot!(punc, extraneuralsol, vars = 2, subplot = sb1, linewidth = lwh + 0.4, color = [:mediumseagreen], 
    linestyle = :dash, label = "NN")

plot!(punc, ts, fitc_mstd[1][i,:], ribbon=fitc_mstd[2][i,:],
        color = :firebrick, fillalpha=.4, linewidth = lwh, label = "FITC", subplot = sb1)
plot!(punc, ts, fitc_mstd[1][i,:] .+ fitc_mstd[2][i,:], label = "", color = :firebrick, linewidth = thinlwh, subplot = sb1)
plot!(punc, ts, fitc_mstd[1][i,:] .- fitc_mstd[2][i,:], label = "", color = :firebrick, linewidth = thinlwh, subplot = sb1)

plot!(punc, ts, fitc_zero_mstd[1][i,:], ribbon=fitc_zero_mstd[2][i,:],
        color = :darkgoldenrod, fillalpha=.4, linewidth = lwh, label = "FITC+{0}", subplot = sb1)
plot!(punc, ts, fitc_zero_mstd[1][i,:] .+ fitc_zero_mstd[2][i,:], label = "", color = :darkgoldenrod, linewidth = thinlwh, subplot = sb1)
plot!(punc, ts, fitc_zero_mstd[1][i,:] .- fitc_zero_mstd[2][i,:], label = "", color = :darkgoldenrod, linewidth = thinlwh, subplot = sb1)

scatter!(punc, datatsteps, ode_data[i,:], subplot = sb1, color = :black, markersize = 3.4, label = "data")  









p = plot(layout = (1,1), size = (400, 380))
ttlftsz = 11
lw = 1.5
sub1 = 1
quiver!(p, getindex.(Zqg,1), getindex.(Zqg,2), quiver = (getindex.(Uqg, 1), getindex.(Uqg,2)), 
    color = Plots.RGBA(0.6, 0.6, 0.6, 0.3), 
    subplot = sub1, linewidth = lw, label = "")

quiver!(p, getindex.(Z,1), getindex.(Z,2), quiver = (getindex.(realU, 1), getindex.(realU,2)), color = :black, subplot = sub1, aspect_ratio = 1,
    title = "", titlefontsize = ttlftsz, linewidth = lw, label = "",)
quiver!(p, getindex.(Z,1), getindex.(Z,2), quiver = (getindex.(fitcU, 1), getindex.(fitcU,2)), color = :firebrick, 
    subplot = sub1, linewidth = lw, label = "")

sub1 = 1
quiver!(p, getindex.(Z,1), getindex.(Z,2), quiver = (getindex.(fitc_zero_U, 1), getindex.(fitc_zero_U,2)), color = :darkgoldenrod, 
    subplot = sub1, linewidth = lw, label = "")

sub1 = 1
quiver!(p, getindex.(Z,1), getindex.(Z,2), quiver = (getindex.(npODEU, 1), getindex.(npODEU,2)), 
    color = Plots.RGBA(0.0, 0.0, 0.8, 0.9), 
    subplot = sub1, linewidth = lw, label = "")

sub1 = 1
quiver!(p, getindex.(Z,1), getindex.(Z,2), quiver = (getindex.(nn_U, 1), getindex.(nn_U,2)), 
    color = Plots.RGBA(0.24, 0.7, 0.44, 0.9),
    subplot = sub1, linewidth = lw, label = "")

plot!(p, xlabel = LaTeXString("\$ x_1 \$"), ylabel = LaTeXString("\$ x_2 \$"), aspect_ratio = 1)




completeplot = plot(p, punc, size = (800, 350), left_margin = ([0 5 5].*Plots.Measures.mm),
    legend = :none )

savefig(completeplot, "../plots/cubedODE.pdf")


#####################################################################################################
## Figure 2
#####################################################################################################

default(tickfont = (9, "serif"))

mksz = 2.1
LTp1 = plot(LTsol, vars = (1,2), color = :black, legend = :none, label = "")
plot!(LTp1, LTgpOde1, vars = (1,2), color = :firebrick, linewidth = 1.8, label = "FITC", legend = :none, aspect_ratio = 1)
plot!(LTp1, xlabel = LaTeXString("\$ x_1 \$"), ylabel = LaTeXString("\$ x_2 \$"))
scatter!(LTp1, ode_data[1,:], ode_data[2,:], color = :black, markersize = mksz, label = "")   


LTp2 = plot(LTsol, vars = (1,2), color = :black, legend = :outerright, label = "")
plot!(LTp2, zeros(2), zeros(2), color = :firebrick, linewidth = 1.8, label = LaTeXString("\\textrm{FITC}"),)
plot!(LTp2, LTgpOde2, vars = (1,2), color = :darkgoldenrod, linewidth = 1.8, label = LaTeXString("\\textrm{FITC+\\{0}\\}"), aspect_ratio = 1)
plot!(LTp2, xlabel = raw"$x_1$", ylabel = raw"$x_2$")
plot!(LTp2, zeros(2), zeros(2), color = :mediumseagreen, linewidth = 1.8, label = LaTeXString("\\textrm{NN}"),)
plot!(LTp2, zeros(2), zeros(2), color = :blue3, linewidth = 1.8, label = LaTeXString("\\textrm{npODE}"),)
plot!(LTp2, zeros(2), zeros(2), color = :black, linewidth = 1.8, label = LaTeXString("\\textrm{truth}"),)
scatter!(LTp2, ode_data[1,:], ode_data[2,:], color = :black, markersize = mksz, label = LaTeXString("\\textrm{data}")) 

LTp3 = plot(LTsol, vars = (1,2), color = :black, legend = :none, label = "")
plot!(LTp3, LTnnODE, vars = (1,2), color = :mediumseagreen, linewidth = 1.5, label = "NN", legend = :none, aspect_ratio = 1)
plot!(LTp3, xlabel = raw"$x_1$", ylabel = raw"$x_2$", aspect_ratio = 1)
scatter!(LTp3, ode_data[1,:], ode_data[2,:], color = :black, markersize = mksz, label = "") 

LTp4 = plot(LTsol, vars = (1,2), color = :black, legend = :none, label = "")
plot!(LTp4, hvars["longtpredh"][:,1], hvars["longtpredh"][:,2], color = :blue3, linewidth = 1.8, label = "npODE", legend = :none, aspect_ratio = 1)
plot!(LTp4, xlabel = raw"$x_1$", ylabel = raw"$x_2$")
scatter!(LTp4, ode_data[1,:], ode_data[2,:], color = :black, markersize = mksz, label = "") 

LTp = plot(LTp1, LTp2, LTp3, LTp4, size = (400, 260), layout = (2,2))

default(tickfont = (10, "serif"))
LTp


savefig(LTp, "../plots/longterm.pdf")