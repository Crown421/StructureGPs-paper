export kmemsample, plotkmemsample, memstats

function kmemsample(traj_sgp, u0, tspan, k, ts; euler::Bool = true, dt = 0.001)
    mgp = kmemoryGP(traj_sgp; k);

    memODE(u, p, t) = mgp(u)

    prob_memODE = ODEProblem(memODE, u0, tspan)

#     memsol = solve(prob_memODE, Tsit5())(ts)
    if euler
        memsol = solve(prob_memODE, Euler(), dt = dt)(ts)
    else
        memsol = solve(prob_memODE, Tsit5())(ts)
    end
end

function memstats(memsols)
    nts = length(memsols[1].t)
    mtrajs = reduce(hcat, [mean(reduce(hcat, [sol.u[k] for sol in memsols]), dims = 2) for k in 1:nts])
    stdtrajs = reduce(hcat, [std(reduce(hcat, [sol.u[k] for sol in memsols]), dims = 2) for k in 1:nts])
    return mtrajs, stdtrajs
end

function plotkmemsample(ts, k10memsols, sol)
    p = plot(layout = (2,2), size = (1000, 500))
    
    for ms in k10memsols[1:8]
        plot!(p, ms.t, getindex.(ms.u, 1), subplot = 1, label = "", title = "$k memory")
        plot!(p, ms.t, getindex.(ms.u, 2), subplot = 3, label = "")
    end

    plot!(p, sol, vars = 1, subplot = 1, label = "truth", color = :black)
    plot!(p, sol, vars = 2, subplot = 3, label = "", color = :black)

    mtrajs10 = reduce(hcat, [mean(reduce(hcat, [sol.u[k] for sol in k10memsols]), dims = 2) for k in 1:nts])
    stdtrajs10 = reduce(hcat, [std(reduce(hcat, [sol.u[k] for sol in k10memsols]), dims = 2) for k in 1:nts])

    plot!(p, ts, mtrajs10[1,:], ribbon=stdtrajs10[1,:],
        color = :salmon1, fillalpha=.4, linewidth = 2.3, subplot = 2)
    plot!(p, sol, vars = 1, color = :black, subplot = 2)
    plot!(p, ts, mtrajs10[2,:], ribbon=stdtrajs10[2,:],
        color = :deepskyblue2, fillalpha=.4, linewidth = 2.3, subplot = 4)
    plot!(p, sol, vars = 2, color = :black, subplot = 4)
    p
end