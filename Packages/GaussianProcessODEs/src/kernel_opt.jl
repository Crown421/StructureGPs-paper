
export train_sparsegp, train
export FITC, VLB


#####
# traditional log-likelihood
#####

function _loglikelihood(logw, sgp::SGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}) where SGP <: SparseGP    
    kernel = sgp.kernel
    X = indP[1]
    Y = indP[2]
    σ_n = sgp.σ_n
    return _loglikelihood(logw, kernel, X, Y, σ_n)
end

function _loglikelihood(logw, kernel, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    K = kernelmatrix(ker, X) + σ_n*I

    Zygote.ignore() do
        if ~isposdef(K)
            println("step")
            println("logw: $logw")
            println("w: $w")
            println("went bad")
        end
    end

    Kchol = cholesky(K)
    
    fitTerm = 1/2 * mapreduce(y -> y' * (Kchol \ y), +, eachrow(vY))
    detTerm = 2* sum(log.(diag(Kchol.L)))
    return fitTerm + detTerm
end


######
# FITC log likelihood cost
######
# add type to SparseGP and then (for NTuple{3,...}) dispatch on it for different likelihoods
function _loglikelihood(logw, sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}} ) where SGP <: SparseGP
    kernel = sgp.kernel
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    σ_n = sgp.σ_n
    _loglikelihood(logw, kernel, Z, X, Y, σ_n)
end

function _loglikelihood(logw, kernel, Z, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Symmetric(Kfu * ( Kuu \ Kfu' ))
    Λ = Diagonal(diag( Kff - Qff) .+ σ_n)
    QS = Qff + Λ

    Zygote.ignore() do
        if ~isposdef(QS)
            println("step")
            println("logw: $logw")
            println("w: $w")
            println("went bad")
        end
    end

    QSChol = cholesky(QS)
    
    nrY = size(vY, 1)
    fitTerm = 1/2 * mapreduce(y -> y' * (QSChol \ y), +, eachrow(vY))
    detTerm = nrY* sum(log.(diag(QSChol.L)))
    return fitTerm + detTerm
end


#####
# gradient, for either of the two above
#####
function _llgrad(G, logw, sgp)
    tmp = gradient(w -> _loglikelihood(logw, sgp, sgp.inP), logw)
    G[:] = tmp[1]
end


######
# VLB, Titsias variatonal lower bound
######
function _variational_lowerbound(logw, sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}) where SGP <: SparseGP
    kernel = sgp.kernel
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    σ_n = sgp.σ_n
    _variational_lowerbound(logw, kernel, Z, X, Y, σ_n)
end

# add type to SparseGP and then (for NTuple{3,...}) dispatch on it for different likelihoods
function _variational_lowerbound(logw, kernel, Z, X, Y, σ_n)
    w = exp.(logw)
    vY = reshape(reduce(vcat, Y), :, length(X)*length(X[1]))
    
    ker = kernel(w)
    Kff = kernelmatrix(ker, X)
    Kfu = kernelmatrix(ker, X, Z)
    Kuu = kernelmatrix(ker, Z)
    
    Qff = Symmetric(Kfu * ( Kuu \ Kfu' ))
    Λ =  σ_n*I
    QS = Qff + Λ

    Zygote.ignore() do
        if ~isposdef(QS)
            println("step")
            println("logw: $logw")
            println("w: $w")
            println("went bad")
        end
    end

    QSChol = cholesky(QS)
    T = Kff - Qff
    
    nrY = size(vY, 1)
    fitTerm = 1/2 * mapreduce(y -> y' * (QSChol \ y), +, eachrow(vY))
    detTerm = nrY * sum(log.(diag(QSChol.L)))
    traceTerm = nrY/(2*σ_n) * tr(T)
    return fitTerm + detTerm + traceTerm
end


function _vlbgrad(G, logw, sgp)
    tmp = gradient(w -> _loglikelihood(logw, sgp, sgp.inP), logw)
    G[:] = tmp[1]
end

# have this dispatch on LL/ elbo object

function define_objective(sgp::SGP; grad = false) where {SGP <: SparseGP, M <: SparseGPMethod}
    return _define_objective(sgp, sgp.inP, sgp.method, grad)
end

function _define_objective(sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}, method::FITC, grad = false) where {SGP <: SparseGP}
    c(logw) = _loglikelihood(logw, sgp, indP)
    if grad
        g(G,logw) = _llgrad(G, logw, sgp)
        return (c,g)
    else
        return c
    end
end

function _define_objective(sgp::SGP, indP::NTuple{3, Array{<:Array{<:Real,1},1}}, method::VLB, grad = false) where {SGP <: SparseGP}
    c(logw) = _variational_lowerbound(logw, sgp, indP)
    if grad
        g(G,logw) = _vlbgrad(G, logw, sgp)
        return (c,g)
    else
        return c
    end
end

function _define_objective(sgp::SGP, indP::NTuple{2, Array{<:Array{<:Real,1},1}}, method, grad = false) where {SGP <: SparseGP}
    # consider warning if non-default method is passed? Stating that it will be ignored?
    c(logw) = _loglikelihood(logw, sgp, indP)
    if grad
        g(G,logw) = _vlbgrad(G, logw, sgp)
        return (c,g)
    else
        return c
    end
end



#####
# training function
#### 
function train_sparsegp(sgp::SGP; 
    show_opt = false, grad = false, options = Optim.Options()) where {SGP <: SparseGP, M <: SparseGPMethod}
    ker = sgp.kernel
    obj = define_objective(sgp; grad = grad)
    optres = optimize(obj, log.(getparam(ker)), options )
    wopt = exp.(optres.minimizer)
    if show_opt
        display(optres)
        display(wopt)
    end
    optker = ker(wopt)
    
    return typeof(sgp)(optker, sgp.σ_n, sgp.inP, sgp.mean, sgp.trafo, sgp.method)
end


function train(gpm::GPM; 
    show_opt = false, grad = false, options = Optim.Options()) where {GPM <: GPmodel, M <: SparseGPMethod}

    sgp = gpm.sgp
    optsgp = train_sparsegp(sgp; show_opt, grad, options)
    return GPmodel(optsgp)
end

function train(gpode::GPO; 
    show_opt = false, grad = false, options = Optim.Options()) where {GPO <: GPODE, M <: SparseGPMethod}

    sgp = gpode.model.sgp
    optsgp = train_sparsegp(sgp; show_opt, grad, options)
    
    return GPODE(GPmodel(optsgp),gpode.tspan,gpode.args...; gpode.kwargs...)
end


