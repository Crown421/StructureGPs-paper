#
export expKernel, rotexpKernel
export computeK # should probably not be exported
export uncoupledMKernel, GIMKernel, rotKernel, KeplerKernel

export keplerKernel

abstract type kerneltype end
# abstract type abstractkronkernel <: kerneltype  end
# abstract type abstracmatrixkernel <: kerneltype end

# temporary
abstract type scalarkernel <: kerneltype  end
abstract type matrixkernel <: kerneltype end

abstract type MatrixKernel <: kerneltype end
abstract type KronMatrixKernel <: kerneltype  end


struct uncoupledMKernel{N, K<:NTuple{N, Kernel}, Qt <: NTuple{N, AbstractArray{<:Real, 2}} } <: KronMatrixKernel 
    kernels::K
    Q::Qt
end

function uncoupledMKernel(K::KT, Q::QT) where {KT<:Kernel, QT <: AbstractArray{<:Real, 2} }
    return uncoupledMKernel((K,), (Q,))
end

function uncoupledMKernel(w::Array{T, 1}) where T<:Real
    K = (pskernel(w),)
    Q = (Diagonal(ones(length(w)-1)),)
    return uncoupledMKernel(K,Q)
end

# evaluate
function (mker::uncoupledMKernel)(z1,z2)
    K = mker.kernels
    Q = mker.Q
    return mapreduce(x -> x[1](z1,z2) * x[2], +, zip(K,Q))
end
    
# reparametrize
function (mker::uncoupledMKernel)(w::Array{<:Real, 1})
    K = mker.kernels
    
    ls = length.(GaussianProcessODEs.getparam.(K))
    endIdx = cumsum(ls)
    startIdx = endIdx .- (ls .- 1)
    ra = range.(startIdx, endIdx, step = 1)
    indw = getindex.(Ref(w), ra)
    
    K = map( (k,w) -> k(w), K, indw)
    typeof(mker)(K, mker.Q)
end




###
# Core structure that is part of each component of the structured matrix kernel
###
struct GIMKernel{GKP, K <: KernelFunctions.Kernel } <: MatrixKernel
    kernel::K
    groupaction::Function
    # may need to be changed for more general group action?
    parameterinterval::Tuple{Float64,Float64}
    gkparams::GKP

    GIMKernel{T,K}(ker::K, grpa, psp, gkparams::T) where {T,K} = new(ker, grpa, psp, gkparams)

end

# constructor
function GIMKernel(ker::K, grpa, parameterinterval, N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, parameterinterval...))
    GIMKernel{typeof(gkparams), K}(ker, grpa, parameterinterval, gkparams)
end

function GIMKernel(ker::K, grpa, parameterinterval, N::Nothing) where K
    GIMKernel{typeof(N), K}(ker, grpa, parameterinterval, N)
end

# evaluation
function (ik::GIMKernel)(z1, z2)
        grpa = ik.groupaction
        gkp = ik.gkparams 
        ker = ik.kernel
    
        base = gkp.weights .* map(x->ker(z1, grpa(x) * z2), gkp.x)
        return sum(base .* grpa.(gkp.x))
        # return sum(ker.(Ref(z1), grpa.(gkp.x) .* [z2]) .* f.(gkp.x) .* gkp.weights)
end


###
# Specialized implementation for rotational equivariant kernel
struct rotKernel{K <: KernelFunctions.Kernel, GKP }  <: MatrixKernel
    kernel::K
    gkparams::GKP
end

# constructor
function rotKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    rotKernel{K, typeof(gkparams)}(ker, gkparams)
end

# group action
function rot(phi) 
    c = cos(phi)
    s = sin(phi)
    [c -s; s c]
end

# evaluation
function (rk::rotKernel)(z1, z2)
    gkp = rk.gkparams 
    ker = rk.kernel

    base = map(x -> ker(z1, rot(x) * z2), gkp.x) .* gkp.weights 
    costerm = sum(base .* cos.(gkp.x) )
    if z1 == z2
        sinterm = 0.
    else
        sinterm = sum(base .* sin.(gkp.x) )
    end
    [costerm -sinterm; sinterm costerm]
end

# reparametize
# there is numerical issues
function (mker::rotKernel)(w::Array{T, 1}) where {T <: Real}
    l = vcat(w[1:2], w[2])
    typeof(mker)(mker.kernel(l), mker.gkparams)
end

function getparam(mker::rotKernel) #where K <: MatrixKernel
    w = getparam(mker.kernel)
    return w[[1,2]]
end



###
# Specialized implementation for kepler system kernel
struct KeplerKernel{K <: KernelFunctions.Kernel, GKP }  <: MatrixKernel
    kernel::K
    gkparams::GKP
end

# constructor
function KeplerKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    KeplerKernel{K, typeof(gkparams)}(ker, gkparams)
end

function KeplerKernel(ker::K, auto) where K
    gkparams = nothing
    KeplerKernel{K, typeof(gkparams)}(ker, gkparams)
end

# group action
keplerrot(phi, x) = vcat(rot(phi)*x[1:2], rot(phi)*x[3:4])

# evaluation
function _computescossin(z1, z2, ker, gkp)
    base = map(x -> ker(z1, keplerrot(x, z2) ), gkp.x) .* gkp.weights 
    costerm = sum(base .* cos.(gkp.x) )
    if z1 == z2
        sinterm = 0.
    else
        sinterm = sum(base .* sin.(gkp.x) )
    end
    return costerm, sinterm
end


function _computescossin(z1, z2, ker, ::Nothing)
    costerm, _ = quadgk(phi -> ker(z1, keplerrot(phi, z2) )* cos(phi), 0, 2*pi, rtol = 1e-8)
    if z1 == z2
        sinterm = 0.
    else
        sinterm, _ = quadgk(phi -> ker(z1, keplerrot(phi, z2) )* sin(phi), 0, 2*pi, rtol = 1e-8)
    end
    return costerm, sinterm
end

function (rk::KeplerKernel)(z1, z2, )
    ker = rk.kernel
    gkp = rk.gkparams

    costerm, sinterm = _computescossin(z1,z2, ker,gkp)
    
    return [costerm -sinterm 0 0;
        sinterm costerm 0 0;
        0 0 costerm -sinterm;
        0 0 sinterm costerm]
end


# reparametize
# there is numerical issues
function (mker::KeplerKernel)(w::Array{T, 1}) where {T <: Real}
    # l = vcat(w[1:2], w[2], w[2], w[2])
    l = vcat(w[1:2], w[2], w[3], w[3])
    typeof(mker)(mker.kernel(l), mker.gkparams)
end

function getparam(mker::KeplerKernel) #where K <: MatrixKernel
    w = getparam(mker.kernel)
    return w[[1,2,4]]
    # return w[[1,2]]
end


### Old stuff


struct expKernel <: scalarkernel 
    param::Array{Float64, 1}
end
function kernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    w = ker.param[2:d+1] .^2  #better without square , very odd, likely issue with forward differences
    return ker.param[1]^2 * exp(-1/2 * wsqeuclidean(z1, z2, 1.0 ./w))
end


struct rotexpKernel <: matrixkernel 
    param::Array{Float64, 1}
    phis::Array{Float64, 1}
    weights::Array{Float64, 1}

    function rotexpKernel(param; N = 30)
        phis, weights = gauss(N, 0, 2*pi)
        new(param, phis, weights)
    end
end


function rotexpIntegrand(phi, x1, x2, w)
    GaussianProcessODEs.kernelfunctionf(x1, rot(phi)* x2, expKernel(w))
end

function kernelfunctionf(z1, z2, ker::rotexpKernel)
    d = length(z1)
    w = ker.param
    # w = vcat(ker.param[1], ker.param[2:d+1] .^2)  
    # costerm, _ = quadgk(phi -> rotexpIntegrand(phi, z1, z2, w)* cos(phi), 0, 2*pi, rtol = 1e-6)
    # sinterm, _ = quadgk(phi -> rotexpIntegrand(phi, z1, z2, w)* sin(phi), 0, 2*pi, rtol = 1e-6)

    phi = ker.phis
    weights = ker.weights

    base = rotexpIntegrand.(phi, Ref(z1), Ref(z2), Ref(w)) .*weights

    costerm = sum(base .* cos.(phi) )
    sinterm = sum(base .* sin.(phi) )

    K = [costerm -sinterm; sinterm costerm]
end

# very ad-hoc, needs much better implementation
struct keplerKernel <: matrixkernel 
    param::Array{Float64, 1}
    phis::Array{Float64, 1}
    weights::Array{Float64, 1}

    function keplerKernel(param; N = 30)
        phis, weights = gauss(N, 0, 2*pi)
        new(param, phis, weights)
    end
end

function keplrotexpIntegrand(phi, x1, x2, w)
    x2r = vcat(rot(phi)*x2[1:2], rot(phi)*x2[3:4])
    GaussianProcessODEs.kernelfunctionf(x1, x2r, expKernel(w))
end
function kernelfunctionf(z1, z2, ker::keplerKernel)
    d = length(z1)
    w = ker.param
    # w = vcat(ker.param[1], ker.param[2:d+1] .^2)  

    phi = ker.phis
    weights = ker.weights

    base1 = keplrotexpIntegrand.(phi, Ref(z1), Ref(z2), Ref(w))
    # base2 = rotexpIntegrand.(phi, Ref(z1[3:4]), Ref(z2[3:4]), Ref(w))

    costerm1 = sum(base1 .* cos.(phi) .*weights)
    sinterm1 = sum(base1 .* sin.(phi) .*weights)

    # costerm2 = sum(base2 .* cos.(phi) .*weights)
    # sinterm2 = sum(base2 .* sin.(phi) .*weights)

    K = [costerm1 -sinterm1 0 0;
         sinterm1 costerm1 0 0;
         0 0 costerm1 -sinterm1;
         0 0 sinterm1 costerm1]
    # K = [0 0 costerm2 -sinterm2;
    #      0 0 sinterm2 costerm2;
    #      costerm1 -sinterm1 0 0;
    #      sinterm1 costerm1 0 0]
end

function dist(z1, z2, ker)
    d = length(z1)
    w = ker.param[2:d+1] .^2    
    L = diagm( ( 1.0 ./ w))
    x = z1 - z2
    ker.param[1]^2 * exp(-1/2 * x'*L*x )
end

function derivativekernelfunctionf(z1, z2, ker::expKernel)
    d = length(z1)
    return ker.param[1]^2 * exp(-1/2 * weuclidean(z1, z2, 1.0 ./ ker.param[2:d+1])^2) * ( (z1.-z2)./ ker.param[2:d+1])
end

# function matrixkernelfunction(z1::Array{Float64, 1}, z2::Array{Float64, 1}, ker::T) where T <: scalarkernel
#     Id = Matrix{Float64}(LinearAlgebra.I, length(z1), length(z2))
#     return kernelfunctionf(z1, z2, ker) * Id
# end
