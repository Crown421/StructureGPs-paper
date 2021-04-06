import KernelFunctions: kernelmatrix

export pskernel, psmkernel
export getparam
###
# simple constructor for a parametric scalar kernel
###
function pskernel(w, ker::K = SqExponentialKernel()) where K <: Kernel
    l = 1 ./ w[2:end]
    w[1]*TransformedKernel(ker,ARDTransform(l))
end

function psmkernel(dims::Int)
    w = ones(dims+1) + 0.4*(rand(dims+1).-0.5) 
    return uncoupledMKernel(pskernel(w), Diagonal(ones(dims)) )
end

function psmkernel(w::Array{T,1}) where T <: Real
    dims = length(w) - 1
    return uncoupledMKernel(pskernel(w), Diagonal(ones(dims)) )
end


#####
# create new kernel with updated parameters
# currently limited functions
#####
function (ker::ScaledKernel)(w)
    typeof(ker)( ker.kernel(w[2:end]),  [w[1]])
end

# should include a check that w is as long as the current parametera
function (ker::TransformedKernel)(w)
    # not sure I need this, (eventually) just used for optimization
    l = 1 ./ w 
#     l = w
    typeof(ker)(ker.kernel, typeof(ker.transform)(l))
end

# ToDO: eventually might need some getfield magic, for recursion. Problem is to know when to stop ( could probably dispatch on SimpleKernel (doesn't have parameters))
# SqExponentialKernel <: KernelFunctions.SimpleKernel > true
# but also, might not want to change the parameters in say gamma exponential kernel


#####
# Obtain scalar kernel parameters
#####
function getparam(x::Array{T, 1}) where {T <: Real}
     return x
end
function getparam(ker::TransformedKernel{K, ARDTransform{Array{Float64,1}}}) where K <: Kernel
    return 1/sqrt(2.) ./ ker.transform.v
end

function getparam(ker)
    res = getparam.(getfield.(Ref(ker), fieldnames(typeof(ker))))
    return reduce(vcat, reverse(res))
end

function getparam(ker::K) where K <: KronMatrixKernel
    kernels = ker.kernels
    params = getparam.(kernels)
    return reduce(vcat, params)
end

function getparam(mker::K) where K <: MatrixKernel
    return getparam(mker.kernel)
end




#######
# compute kernel matrices
#######

###
# for Kronecker compatible kernels

function kernelmatrix(mker::K, Z) where K<:KronMatrixKernel 
    # remarkably faster than mapreduce( (k,q) -> kron(kernelmatrix(k, test), q), +, K,Q)
    KE = mker.kernels
    Q = mker.Q
    mapreduce(x -> kron(kernelmatrix(x[1], Z), x[2]), +, zip(KE,Q))
end

function kernelmatrix(mker::K, a, b) where K<:KronMatrixKernel 
    # remarkably faster than mapreduce( (k,q) -> kron(kernelmatrix(k, test), q), +, K,Q)
    KE = mker.kernels
    Q = mker.Q
    mapreduce(x -> kron(kernelmatrix(x[1], a, b), x[2]), +, zip(KE,Q))
end


###
# for kernels that return a (full) matrix
function kernelmatrix(mker::K, Z) where K <:MatrixKernel
    innerN= size(Z[1], 1)
    outerN = length(Z)
    n = innerN* outerN
    B = Zygote.Buffer([Z[1][1]], n,n)
    @inbounds for j in 1:outerN
        modj = 1 + (j-1) * innerN
        for i in j:outerN
            modi = 1 + (i-1) * innerN
            B[modi:modi+innerN-1, modj:modj+innerN-1] = mker(Z[i], Z[j])
        end
        for i in 1:j-1
            modi = 1 + (i-1) * innerN
            B[modi:modi+innerN-1, modj:modj+innerN-1] = B[modj:modj+innerN-1, modi:modi+innerN-1]'
        end
    end
    return copy(B)
end


function kernelmatrix(mker::K, a, b) where K <:MatrixKernel
    innerN= size(a[1], 1)
    outerN1 = length(a)
    outerN2 = length(b)
    n1 = innerN* outerN1
    n2 = innerN* outerN2
    B = Zygote.Buffer([a[1][1]], n1,n2)
    @inbounds for j in 1:outerN2
        modj = 1 + (j-1) * innerN
        for i in 1:outerN1
            modi = 1 + (i-1) * innerN
            B[modi:modi+innerN-1, modj:modj+innerN-1] = mker(a[i], b[j])
        end
    end
    return copy(B)
end




####
## Old
###


function computeK(Z, kernelT::S) where {S <:matrixkernel }
    tmp = pairwise((z1, z2) -> kernelfunctionf(z1, z2, kernelT), Z,  Symmetric)
    K = symblockreduce(tmp)
end

function computeK(a, b, kernelT::S) where {S <:matrixkernel }
    tmp = [GaussianProcessODEs.kernelfunctionf(z1, z2, kernelT) for z1 in a, z2 in b]
    K = blockreduce(tmp)
end

function computeK(Z, kernelT::S) where {S <: scalarkernel}
    tmp = pairwise((z1, z2) -> kernelfunctionf(z1, z2, kernelT), Z,  Symmetric)
    x = Z[1]
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    kron(tmp, Id)
end

function computeK(a, b, kernelT::S) where {S <: scalarkernel}
    tmp = [GaussianProcessODEs.kernelfunctionf(z1, z2, kernelT) for z1 in a, z2 in b]
    x = a[1]
    Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
    kron(tmp, Id)
end

#TODO add matrix valued version
# might have to take the symblock approach, allocate whole matrix and fill block by block

function computeK(a, kernelT)
    computeK(a, a, kernelT)
end

# TODO might be obsolete
# function Kx(x, Z, ker::T) where T <: scalarkernel
#     Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))
#     scalars = kernelfunctionf.(Ref(x), Z, Ref(ker))
#     (kron(scalars', Id))
# end

# function Kx(x, Z, ker::T) where T <: matrixkernel
#     computeK([x], Z, ker)
# end


# function dKx(x, Z, ker::T) where T <: scalarkernel
#     Id = Matrix{Float64}(LinearAlgebra.I, length(x), length(x))

#     tmp = GaussianProcessODEs.derivativekernelfunctionf.(Ref(x), Z, Ref(ker))
#     tmp = reduce(hcat, tmp)
#     (kron(tmp, Id))
# end


###
# reduce matrix of arrays into blockmatrix
###

# export symblockreduce
# this one might/should be more efficient due to 
function symblockreduce(A)
    innerN = size(A[1], 1)
    outerN = size(A, 1)
    n = innerN* outerN
    B = Symmetric(Matrix{typeof(A[1][1])}(undef, n, n))
    # B = Matrix{typeof(A[1][1])}(undef, n, n)
    # @inbounds
    for j in 1:outerN
        modj = 1 + (j-1) * innerN
        for i in 1:outerN
            modi = 1 + (i-1) * innerN
            B.data[modi:modi+innerN-1, modj:modj+innerN-1] .= A[i,j]
        end
    end
    return B
end

function blockreduce(A)
    reduce(vcat, [reduce(hcat, A[i, :]) for i in 1:size(A, 1)])
end