export crotinvKernel, Kx
export dKernel, dcrotinvKernel

# TODO: combine with other group action (currently f(phi,x) not rot(phi)*x)
function Krot(phi) 
    c = cos(phi)
    s = sin(phi)
    [c -s 0 0; s c 0 0; 0 0 c -s; 0 0 s c]
end

### invariant Kernel
struct crotinvKernel{K <: KernelFunctions.Kernel, GKP }  <: MatrixKernel
    kernel::K
    gkparams::GKP
end

# constructor
function crotinvKernel(ker::K; N::Int) where K
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
#     gkparams = zip(gkparams.x, gkparams.weights)
#     gkparams = Iterators.product(gkparams, gkparams)
    crotinvKernel{K, typeof(gkparams)}(ker, gkparams)
end

function (crker::crotinvKernel)(x1::Array{T,1},x2::Array{T,1}) where T <: Real
    phis = crker.gkparams.x
    
    rx1 = Krot.(phis) .* [x1]
    rx2 = Krot.(phis) .* [x2]
    
    return crker(rx1, rx2)
end

function (crker::crotinvKernel)(x1::Array{T,1},rx2::Array{Array{T,1},1}) where T <: Real
    phis = crker.gkparams.x
    
    rx1 = Krot.(phis) .* [x1]
    
    return crker(rx1, rx2)
end

function (crker::crotinvKernel)(rx1::Array{Array{T,1},1}, rx2::Array{Array{T,1},1}) where T <: Real
    w = crker.gkparams.weights
    ker = crker.kernel
    iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    tmp = mapreduce(iI -> w[iI[1]] * ker(rx1[iI[1]], rx2[iI[2]]) * w[iI[2]], + , iterIdx)
    return tmp
end

struct Kx{K, rZ}
    rker::K
    rotZ::rZ
end

function Kx(ker::K, Z) where K <: MatrixKernel
    phis = ker.gkparams.x
    tmpRots = Krot.(phis)
    rotZ = [tmpRots .* [z] for z in Z]
    
    Kx{K, typeof(rotZ)}(ker, rotZ)
end

function (Kx::Kx)()
    rZ = Kx.rotZ
#     w = Kx.rker.gkparams.weights
#     ker = Kx.rker.kernel
#     iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    N = length(Kx.rotZ)
    K = zeros(N, N)
    
    Threads.@threads for i in 1:N
        for j in 1:i
            tmp = Kx.rker(rZ[i], rZ[j])
#             tmp = mapreduce(iI -> w[iI[1]] * ker(rZ[i][iI[1]], rZ[j][iI[2]]) * w[iI[2]], + , iterIdx)
            K[i,j] = K[j,i] = tmp
        end
    end
    return K
end

function (Kx::Kx)(x)
    rZ = Kx.rotZ
#     w = Kx.rker.gkparams.weights
    ker = Kx.rker.kernel
#     iterIdx = Iterators.product(1:length(w), 1:length(w))
    
    # phis = Kx.rker.gkparams.x
    # rx = Krot.(phis) .* [x]
    
    D = length(ker(x,x))
    
    N = length(Kx.rotZ)
    K = Zygote.Buffer([0.], D,N)
#     zeros(1, N)
    
#     Threads.@threads for i in 1:N
    for i in 1:N
#         tmp = mapreduce(iI -> w[iI[1]] * ker(rx[iI[1]], rZ[i][iI[2]]) * w[iI[2]], + , iterIdx)
        tmp = Kx.rker(x, rZ[i])
        for (j, val) in enumerate(tmp)
            K[j,i] = val
        end
#         display(K[:, i])
    end
    return copy(K)
end


# some extra stuff
function (mker::crotinvKernel)(w::Array{T, 1}) where {T <: Real}
    typeof(mker)(mker.kernel(w), mker.gkparams)
end

function (mker::crotinvKernel)(x1, x2)
    typeof(mker)(mker.kernel(w), mker.gkparams)
end

import KernelFunctions: kernelmatrix
function kernelmatrix(ker::crotinvKernel, X)
    tmpKx = Kx(ker, X)  
    return tmpKx()
end


####################################################
### The derivative kernels
## The base kernel
struct dKernel{dK} <: KernelFunctions.Kernel
    dkernel::dK
end

# constructor
function dKernel(ker::K) where K <: KernelFunctions.Kernel
    dker(x1,x2) = gradient(x1->ker(x1,x2), x1)
    dKernel{typeof(dker)}(dker)
end

function (dker::dKernel)(x1,x2)
    dker.dkernel(x1,x2)[1]
end


## the invariant derivative kernel 
struct dcrotinvKernel{K <: KernelFunctions.Kernel, GKP, DI }  <: MatrixKernel
    kernel::K
    gkparams::GKP
    derividx::DI
end
# TODO: possibly make things a little more efficient by only computing part of the kernel

function dcrotinvKernel(ker::K; N::Int, di=:) where K <: dKernel
    gkparams = NamedTuple{(:x, :weights)}(gauss(N, 0, 2pi))
    dcrotinvKernel{K, typeof(gkparams), typeof(di)}(ker, gkparams, di)
end

function (crker::dcrotinvKernel)(x1::Array{T,1},x2::Array{T,1}) where T <: Real
    phis = crker.gkparams.x
    R = GaussianProcessODEs.Krot.(phis)
    rx1 = R .* [x1]
    rx2 = R .* [x2]
    return crker(R, rx1, rx2)
end

function (crker::dcrotinvKernel)(x1::Array{T,1},rx2::Array{Array{T,1},1}) where T <: Real
    phis = crker.gkparams.x
    R = GaussianProcessODEs.Krot.(phis)
    rx1 = R .* [x1]
    return crker(R, rx1, rx2)
end

function _rh(R, di)
    transpose(R)[di, :]
end

function (crker::dcrotinvKernel)(R, rx1::Array{Array{T,1},1}, rx2::Array{Array{T,1},1}) where T <: Real
    w = crker.gkparams.weights
    dker = crker.kernel
    di = crker.derividx
    iterIdx = Iterators.product(1:length(w), 1:length(w))
#     
    tmp = mapreduce(iI -> w[iI[1]] * _rh(R[iI[1]], di) * dker(rx1[iI[1]], rx2[iI[2]]) * w[iI[2]], + , iterIdx)
    return tmp
end