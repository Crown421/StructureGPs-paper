using GaussianProcessODEs
using UncertainODESolvers


using DifferentialEquations
using Flux, DiffEqFlux
using Optim

using Clustering
using LinearAlgebra

using JLD2

RERUN_OPT = false

# ODE functions
include("functions_kepler.jl")



datasize = 75

q0 = [5., 0.]
x0 = [q0[1]; q0[2]; 0.; 17.]
tspan = (0., 20.)
datatspan = (0., 5.)
datatsteps = range(datatspan[1], datatspan[2], length = datasize)

prob = ODEProblem(ckepler!, x0, tspan, kh)
sol = solve(prob, ImplicitMidpoint(); dt = 0.01);

ode_data = Array(sol(datatsteps))
traj = sol(datatsteps);


# training the NeuralODE

if RERUN_OPT
    prob_neuralode = NeuralODE(dudt2, datatspan, Tsit5(), saveat = datatsteps)

    function predict_neuralode(p)
    Array(prob_neuralode(x0, p))
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
    pf = result_neuralode2.minimizer;
else
    println("Using pre-optimized NN")
    @load "data/NN_Kepler_parameter.jld2" pf
end

extraneuralprob = NeuralODE(dudt2, tspan, ImplicitMidpoint(); dt = 0.001)
extraneuralsol = extraneuralprob(x0, pf);


# get gradient observations
traj = sol(datatsteps)
data = gradient_data(traj; show_opt = true);

inDim = length(data.X[1]);

# project into quotient space
phis = -atan.(getindex.(data.X,2), getindex.(data.X,1))
rotX = map(GaussianProcessODEs.keplerrot, phis, data.X);
rotY = map(GaussianProcessODEs.keplerrot, phis, data.Y);

rdata = (X = rotX, Y = rotY);

# inducing points
n1 = 10
n34 = 15

x1r = collect(range(4.5, 13, length = n1))
x34r = permutedims(hcat(getindex.(rdata.X, 3), getindex.(rdata.X, 4)))
x34r = (kmeans(x34r, n34).centers)
# g34points = [x[:] + 0.3*f(2) for x  in eachcol(g34points)]
x34r = [x[:] for x  in eachcol(x34r)]

Zcl = [ vcat( [x, 0.0], x34) for x in x1r, x34 in x34r][:];


# Standard gp
let 
    n1 = 35
    n34 = 35

    rotU = rdata.X

    dpoints = permutedims(hcat(getindex.(rotU, 1), getindex.(rotU, 2)))
    gpoints = (kmeans(dpoints, n1).centers)
    gpoints = [x[:] for x  in eachcol(gpoints)]

    d34points = permutedims(hcat(getindex.(rotU, 3), getindex.(rotU, 4)))
    g34points = (kmeans(d34points, n34).centers)
    # g34points = [x[:] + 0.4*f(2) for x  in eachcol(g34points)]
    g34points = [x[:] for x  in eachcol(g34points)]

    Z = [ vcat( x, x34) for x in gpoints, x34 in g34points][:]
    display(length(Z))

    ### barely tractable optimization problem, the values below required multiple restarts 
    ### (with random initial values)

    if RERUN_OPT
        ## to optimize yourself, run the following
        kker2 = psmkernel(4)
        traj_sgp2 = SparseGP(kker2, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-5);
        gpOde2 = GPODE(GPmodel(traj_sgp2), tspan, ImplicitMidpoint(); dt = 0.01);

        options2 = Optim.Options(iterations = 1000, g_tol = 1e-5, show_trace = true, show_every = 200)
        res_gpODE = train(gpOde2; show_opt = true, options = options2);
    else
        println("Using pre-optimized standard kernel")
        kker2 = psmkernel([923.96750167, 580.11637136, 1.52809373111, 12.5317653133, 10.4346816])
        traj_sgp2 = SparseGP(kker2, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-5);
        res_gpODE = GPODE(GPmodel(traj_sgp2), tspan, ImplicitMidpoint(); dt = 0.01);
    end

    

    global gpsol = res_gpODE(x0)
end


# Equivariant kernel
if RERUN_OPT
    kker = KeplerKernel(pskernel(ones(5)), N = 70)

    traj_sgp = SparseGP(kker, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-4);
    gpOde = GPODE(GPmodel(traj_sgp), tspan, ImplicitMidpoint(); dt = 0.01);

    options = Optim.Options(iterations = 60, g_tol = 1e-6, show_trace = true, show_every = 20)
    res_gpODE = train(gpOde; show_opt = true, options = options);
else
    println("Using pre-optimized equivariant kernel")
    kker = KeplerKernel(pskernel([518.06211, 2.07969, 2.07969, 58.36188, 58.36188]), N = 70)
    traj_sgp = SparseGP(kker, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-4);
    res_gpODE = GPODE(GPmodel(traj_sgp), tspan, ImplicitMidpoint(); dt = 0.01);
end

optgpsol = res_gpODE(x0);


###  low accuracy integration
optpar = getparam(res_gpODE.model.sgp.kernel)
Nkker = KeplerKernel(pskernel(optpar[[1,2,2,3,3]]), N = 30)
NgpODE = GPODE(GPmodel(SparseGP(Nkker, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-4)), 
    tspan, ImplicitMidpoint(); dt = 0.01);

Noptsol = NgpODE(x0);


### second order system

o2data = (X=map( x -> x[1:2], data.X), Y=map( x -> x[3:4], data.Y));

phis = -atan.(getindex.(o2data.X,2), getindex.(o2data.X,1))
rotX = map((phi, x) -> GaussianProcessODEs.rot(phi) * x, phis, o2data.X);
rotY = map((phi, x) -> GaussianProcessODEs.rot(phi) * x, phis, o2data.Y);
ro2data = (X = rotX, Y = rotY);

# grid for second order
n = 10
x1range = range(minimum(getindex.(ro2data.X, 1)), maximum(getindex.(ro2data.X, 1)), length = n) 
Zcl2o = [ [x, 0.0] for x in x1range][:];

# optimization
if RERUN_OPT
    rker = rotKernel(pskernel([100., 2., 2.]), N = 70)
    options = Optim.Options(iterations = 550, g_tol = 1e-7, show_trace = true, show_every = 50)
    VLB_traj2o_sgp = SparseGP(rker, Zcl2o, ro2data.X, ro2data.Y; method = VLB());
    VLB_gpmodel2o = train(GPmodel(VLB_traj2o_sgp); show_opt = true, options = options);

else
    rker = rotKernel(pskernel([587.87688, 2.61006, 2.61006]), N = 70)
    VLB_traj2o_sgp = SparseGP(rker, Zcl2o, ro2data.X, ro2data.Y; method = VLB())
    VLB_gpmodel2o = GPmodel(VLB_traj2o_sgp)
end

q0 = x0[1:2]
p0 = x0[3:4]


VLBprob = DynamicalODEProblem(pdot, qdot, p0, q0, tspan, VLB_gpmodel2o)
VLB_gp2osol = solve(VLBprob, KahanLi6(), dt=1//10);


#########
# long term solve

println("Starting long term solve")
# low accuracy equivariant GP
longtspan = (0., 120.)
longNgpODE = GPODE(GPmodel(SparseGP(Nkker, Zcl, rdata.X, rdata.Y; method = FITC(), σ_n = 1e-4)), 
    longtspan, ImplicitMidpoint(); dt = 0.01);
longNoptsol = longNgpODE(x0);

# equivariant GP
longgpOde = GPODE(res_gpODE.model, longtspan, ImplicitMidpoint(); dt = 0.01);
longoptgpsol = longgpOde(x0);

# second order system
longVLBprob = remake(VLBprob; tspan=longtspan)
longVLB_gp2osol = solve(longVLBprob, KahanLi6(), dt=1//100);
longVLB_gp2osol = (t=longVLB_gp2osol.t, u=[ vcat.(u...)[[3,4,1,2]]  for u in longVLB_gp2osol.u])



@save "data/kepler.jld2" sol optgpsol Noptsol VLB_gp2osol gpsol extraneuralsol traj longNoptsol longoptgpsol longVLB_gp2osol