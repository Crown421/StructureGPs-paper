using GaussianProcessODEs
using UncertainODESolvers
# using BenchmarkTools
# using KernelFunctions

using ThreadsX

using DifferentialEquations
using Flux, DiffEqFlux

using Optim

using LinearAlgebra

using MAT
using JLD2


u0 = Float32[2.0; 0.0]
datasize = 20
tspan = (0.0f0, 3.0f0)
datatspan = (0.0f0, 1.5f0)
datatsteps = range(datatspan[1], datatspan[2], length = datasize)

function trueODEfunc!(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
function trueODEfunc(u,p,t)
    du = similar(u)
    trueODEfunc!(du, u, p, t)
end

prob_trueode = ODEProblem(trueODEfunc!, u0, tspan)
# ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
sol = solve(prob_trueode, Tsit5())
ode_data = Array(sol(datatsteps))

traj = sol(datatsteps);





let
    global dudt2 = FastChain((x, p) -> x.^3,
                      FastDense(2, 50, tanh),
                      FastDense(50, 2))
    prob_neuralode = NeuralODE(dudt2, datatspan, Tsit5(), saveat = datatsteps)

    function predict_neuralode(p)
      Array(prob_neuralode(u0, p))
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, ode_data .- pred)
        return loss, pred
    end


    result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                              ADAM(0.05),
                                              maxiters = 300)

    result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                               result_neuralode.minimizer,
                                               DiffEqFlux.LBFGS())
    # display(result_neuralode2)
    global pf = result_neuralode2.minimizer;

    global extraneuralprob = NeuralODE(dudt2, tspan, Tsit5())
    global extraneuralsol = extraneuralprob(u0, pf);
end;



traj = sol(datatsteps)
data = gradient_data(traj; show_opt = true);

borders = getborders(data)
ngp = 4
Z = rectanglegrid(borders, ngp)

inDim = length(data.X[1]);




nts = 110
ts = range(tspan..., length = nts)
k = 80# memory size
n = 150
eulerdt = 0.0001


### FITC computations
# basic
let
    mker = psmkernel([779.6017752888342, 2.6381334074291454, 2.6159588118755073])
    traj_sgp = SparseGP(mker, Z, data.X, data.Y; σ_n = 1e-9)
    
    global fitc_traj_sgp = train_sparsegp(traj_sgp; show_opt = true)
    global fitc_gpODE = GPODE(fitc_traj_sgp, tspan, Tsit5())    
    global fitc_gpsol = fitc_gpODE(u0)  

    @time memsols = ThreadsX.map(i->kmemsample(fitc_traj_sgp, u0, tspan, k, ts; euler = true, dt = eulerdt), 1:n)
    global fitc_mstd = memstats(memsols)
end;

# with the added zero
let
    ndata = (X=vcat(data.X, [[0f0, 0f0]]), Y=vcat(data.Y, [[0., 0.]]) )
    
    mker = psmkernel([1456.6333489090493, 2.965623835914896, 2.969681882287084])
    traj_sgp = SparseGP(mker, Z, ndata.X, ndata.Y;  σ_n = 1e-9)#, method = FITC())

    options = Optim.Options(iterations = 1000, g_tol = 1e-6, show_trace = true, show_every = 500)
    global fitc_zero_traj_sgp = train_sparsegp(traj_sgp; show_opt = true, options = options)
    global fitc_zero_goODE = GPODE(fitc_zero_traj_sgp, tspan, Tsit5()) 
    global fitc_zero_gpsol = fitc_zero_goODE(u0) 

    @time memsols = ThreadsX.map(i->kmemsample(fitc_zero_traj_sgp, u0, tspan, k, ts; euler = true, dt = eulerdt), 1:n)
    global fitc_zero_mstd = memstats(memsols)
end;


# import data from heinonen script
hvars = matread("data/heinonen_data.mat")

# really needs a revamp
let
    train = sum(norm.(sol(datatsteps).u .- fitc_gpsol(datatsteps).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(datatsteps).u .- fitc_zero_gpsol(datatsteps).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(datatsteps).u .- extraneuralsol(datatsteps).u)) / length(datatsteps)
    display(train)
    npodeSol = [u[:] for u in eachrow(hvars["errorXh"])]
    train = sum(norm.(sol(datatsteps).u .- npodeSol[1:20])) / length(datatsteps)
    display(train)
    
    display("-----------")
    testtime = range(datatspan[end], tspan[end], step = datatsteps.step)
    train = sum(norm.(sol(testtime).u .- fitc_gpsol(testtime).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(testtime).u .- fitc_zero_gpsol(testtime).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(testtime).u .- extraneuralsol(testtime).u)) / length(datatsteps)
    display(train)
    npodeSol = [u[:] for u in eachrow(hvars["errorXh"])]
    train = sum(norm.(sol(testtime).u .- npodeSol[20:end])) / length(datatsteps)
    display(train)
    
    display("-----------")
    testtime = range(datatspan[1], tspan[end], step = datatsteps.step)
    train = sum(norm.(sol(testtime).u .- fitc_gpsol(testtime).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(testtime).u .- fitc_zero_gpsol(testtime).u)) / length(datatsteps)
    display(train)
    train = sum(norm.(sol(testtime).u .- extraneuralsol(testtime).u)) / length(datatsteps)
    display(train)
    npodeSol = [u[:] for u in eachrow(hvars["errorXh"])]
    train = sum(norm.(sol(testtime).u .- npodeSol[1:end])) / length(datatsteps)
    display(train)
    
#     fulltime .- hvars["errortsh"][:]
end


# data for the quiver plot
function normarrow!(U)
    U[:] = U ./ (norm.(U)) .* log10.(norm.(U)) #.* 0.99;
#     U[:] = U ./ maximum(norm.(U))
end

realU = trueODEfunc.(Z, 0, 0);
normarrow!(realU)

fitcU = fitc_gpODE.model.(Z);
normarrow!(fitcU)

fitc_zero_U = fitc_zero_goODE.model.(Z);
normarrow!(fitc_zero_U)
npODEU = [u[:] for u in eachrow(hvars["Uh"])];
normarrow!(npODEU)

nn_U = dudt2.(Z, Ref(pf))
normarrow!(nn_U);

Zqg = rectanglegrid([[-2.5, 2.5],[-2.5, 2.5]], 9)
Uqg = trueODEfunc.(Zqg, 0, 0)
normarrow!(Uqg);



## Data for long term plot

LTtspan = (0., 25.)
LTprob_trueode = ODEProblem(trueODEfunc!, u0, LTtspan)
# ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
LTsol = solve(LTprob_trueode, Tsit5())

LTgpOde1 = GPODE(fitc_traj_sgp, LTtspan, Tsit5())(u0)
LTgpOde2 = GPODE(fitc_zero_traj_sgp, LTtspan, Tsit5())(u0)
# LTgpOde3 = GPODE(FITC_cub_goOde.model.sgp,LTtspan, Tsit5())(u0)
# LTgpOde4 = GPODE(FITC_cub_zero_goOde.model.sgp, LTtspan, Tsit5())(u0);

longneuralprob = NeuralODE(dudt2, LTtspan, Tsit5())
LTnnODE = longneuralprob(u0, pf);



@save "data/spiral.jld2" sol hvars extraneuralsol fitc_mstd fitc_zero_mstd datatsteps ode_data Z fitc_gpODE fitc_zero_goODE LTsol LTgpOde1 LTgpOde2 LTnnODE realU fitcU fitc_zero_U npODEU nn_U Zqg Uqg ts