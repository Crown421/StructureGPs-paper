export SparseGP, GPmodel, GPODE
export samplegp, gpmeanvar

# https://github.com/SciML/DiffEqFlux.jl/blob/c59971fd4d3ee84aff39f88b7073d7e8cf51c34c/src/neural_de.jl#L38

abstract type TrajType end
struct SampleTraj <: TrajType end
struct MeanTraj <: TrajType end

abstract type SparseGPMethod end
struct FITC <: SparseGPMethod end
struct VLB <: SparseGPMethod end
struct Vanilla <: SparseGPMethod end

###
# Struct that contains everything needed for prediction
###
struct SparseGP{K, T<:Real, N, A<: NTuple{N, Array{<:Array{<:Real,1},1}}, M<:SparseGPMethod}
    kernel::K
    σ_n::T
    inP::A
    mean::Function
    trafo::Function
    method::M
    # type? FITC, SOR, PITC
end

#ToDo: Update σ_n in this object

function zeromean(n) 
    return x -> fill(0, n)
end
identitytrafo(x) = x

function SparseGP(kernel, Z, U; σ_n = 1e-6, mean = zeromean(length(U[1])), trafo = identitytrafo, method = Vanilla())
    indP = (trafo.(Z), U)
    N = length(indP)
    SparseGP{typeof(kernel), typeof(σ_n), N, typeof(indP), typeof(method)}(kernel, σ_n, indP, mean, trafo, method)
end
function SparseGP(kernel, Z, X, Y; σ_n = 1e-6, mean = zeromean(length(Y[1])), trafo = identitytrafo, method = FITC())
    indP = (trafo.(Z), trafo.(X), Y)
    N = length(indP)
    SparseGP{typeof(kernel), typeof(σ_n), N, typeof(indP), typeof(method)}(kernel, σ_n, indP, mean, trafo, method)
end

function (sgp::SparseGP)(x::T) where T <: Real
    Z = sgp.inP[1]
    ker = sgp.kernel
    Kx = kernelmatrix(ker, [[x]], Z)
end

function (sgp::SparseGP)(x::Array{T,1}) where T <: Real
    Z = sgp.inP[1]
    ker = sgp.kernel
    Kx = kernelmatrix(ker, [x], Z)
end



###
# GP model, that contains the sparse GP object with all necessary data, and 
struct GPmodel{SGP <: SparseGP, T<:Real}
    sgp::SGP
    KinvU::Array{T,2}
    Σ::Array{T,2}
end 

# maybe with data
function GPmodel(sgp::SparseGP)
    KiU, Σ = computeKinvU(sgp)
    GPmodel(sgp, KiU, Σ)
end

function (gpm::GPmodel)(x, traj::MeanTraj = MeanTraj())
    Kx = gpm.sgp(gpm.sgp.trafo(x))
    μ = gpm.sgp.mean
    return (μ(x) .+ (Kx * gpm.KinvU)[:])
end

function (gpm::GPmodel)(x, traj::SampleTraj)
    Kx = gpm.sgp(gpm.sgp.trafo(x))
    μ = gpm.sgp.mean
    m =  (μ(x) .+ (Kx * gpm.KinvU))[:]
    
    Σ = gpm.Σ

    std = sqrt.(diag(Kx * (Σ \ Kx')))
    # var = diag(Kx*Kuu*Kx')
    s = randn(length(x)) .* std
    return (m .+ s)
end

function (gpm::GPmodel)(xv::MS) where MS  <: Array{<:Measurement{<:Real}, 1}
    x = getfield.(xv, :val)
    Kx = gpm.sgp(gpm.sgp.trafo(x))
    μ = gpm.sgp.mean
    m =  (μ(x) .+ (Kx * gpm.KinvU))[:]
    
    Σ = gpm.Σ

    var = sqrt.(diag(Kx * (Σ \ Kx')))
    # var = diag(Kx*Kuu*Kx')

    return (m .± var)
end


####
# functions to facilitate efficient computation, as per Q-C&R
function computeKinvU(sgp::SparseGP)
    return _computeKinvU(sgp, sgp.inP, sgp.method)
end

function _computeKinvU(sgp::SparseGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}, method::M) where M<:SparseGPMethod 
    Z = indP[1]
    U = indP[2]
    σ_n = sgp.σ_n
    μ = sgp.mean
    U = U .- μ.(Z)
#     vU = reduce(vcat, U)
    # ToDo: might have to make output(?) dimensions more explicit
    vU = reshape(reduce(vcat, U), :, length(Z)*length(Z[1]))
    vU = permutedims(vU)
    ker = sgp.kernel
    K = kernelmatrix(ker, Z) + σ_n * I
    KinvU = K \ vU
    return (KinvU, K) 
end

function _computeKinvU(sgp::SparseGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}, method::M) where M <: SparseGPMethod 
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    μ = sgp.mean
    Y = Y .- μ.(X)
    ker = sgp.kernel
    
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Kfu * ( Kuu \ Kfu' )
    
    noise = sgp.σ_n
    Λ = _computelambda(Kff, Qff, noise, method)
    
    Σ = _computesigma(Kuu, Kfu, Λ)
    
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    vY = permutedims(vY)
    KinvU = Σ \ (Kfu' * (Λ \ vY))
    return (KinvU, Σ)
end

function _computesigma(Kuu, Kfu, Λ)
    return Kuu + Kfu' * (Λ \ Kfu)
end

function _computelambda(Kff, Qff, noise, method::VLB)
    return noise * I
end

function _computelambda(Kff, Qff, noise, method::FITC)
    return Diagonal(diag( Kff - Qff) .+ noise)
end

#####
# complete GPODE construct
#####
basic_tgrad(u,p,t) = zero(u)
struct GPODE{M<:GPmodel,T,A,K,TT<: TrajType} #<: NeuralDELayer
    model::M
    # p::P, parameters, maybe one day
    tspan::T
    args::A
    kwargs::K
    trajtype::TT

    function GPODE(model::GPM,tspan,args...; sample = false, kwargs...) where GPM <: GPmodel
        tt = sample ? SampleTraj() : MeanTraj()
        new{typeof(model),typeof(tspan),typeof(args),typeof(kwargs), typeof(tt)}(
            model,tspan,args,kwargs,tt)
    end
end

function GPODE(sgp::SGP,tspan,args...;sample::Bool=false, kwargs...) where SGP <: SparseGP
    gpm = GPmodel(sgp)
    return GPODE(gpm,tspan,args..., sample = sample, kwargs...)
end

function (gp::GPODE)(x0)
    dudt_(u,p,t) = gp.model(u, gp.trajtype)
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x0,getfield(gp,:tspan))
    solve(prob,gp.args...;gp.kwargs...)
end


function samplegp(x, gp, n, args = gp.args; kwargs...)
    #     dudt_(u,p,t) = gp.model(u, GaussianProcessODEs.SampleTraj())
    #     ff = ODEFunction{false}(dudt_,tgrad=GaussianProcessODEs.basic_tgrad)
    #     prob = ODEProblem{false}(ff,x,getfield(gp,:tspan))
    #     ensemble = EnsembleProblem(prob)
    #     solve(ensemble, gp.args..., EnsembleThreads(); trajectories = n, gp.kwargs...)
        
        sample_gpODE = GPODE(gp.model, gp.tspan, 
            args...; sample = true, kwargs...);
        t = @elapsed sols = ThreadsX.collect(sample_gpODE(x) for i in 1:n)
        EnsembleSolution(sols, t, true)
end


using DifferentialEquations.EnsembleAnalysis
function gpmeanvar(samplesols, ts)
    meanvar = timepoint_meanvar(samplesols,ts)
    mtrajs = meanvar[1]
    stdtrajs = sqrt.(meanvar[2])
    return mtrajs, stdtrajs
end