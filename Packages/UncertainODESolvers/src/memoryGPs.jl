export kmemoryGP


mutable struct kmemoryGP{K, T<:Real}
    kernel::K
    memsize::Int
    iter::Base.Iterators.Stateful{Base.Iterators.Cycle{UnitRange{Int64}},Union{Nothing, Tuple{Int64,Int64}}}
    baseZ::Array{Array{T, 1},1}
    Z::Array{Array{T, 1},1}
    baseU::Array{Array{T, 1},1}
    U::Array{Array{T, 1},1}
    BaseKchol::Cholesky{T,Array{T,2}}
    Kchol::Cholesky{T,Array{T,2}}
    σ_n::T
    σ_m::T
end


function kmemoryGP(sgp::GPODEs.SparseGP; k = 1, σ_m = 0.)
    Kchol, U = computeParts(sgp)
    Z = deepcopy(sgp.inP[1])
    σ_n = sgp.σ_n
    kernel = sgp.kernel
    iter = Iterators.Stateful(Iterators.cycle(1:k))
    kmemoryGP{typeof(kernel), typeof(σ_n)}(kernel, k, iter, Z, Z, U, U, Kchol, Kchol, σ_n, σ_m)
end


function (mGP::kmemoryGP)(x)
    ker = mGP.kernel
    Z = mGP.Z
    Kchol = mGP.Kchol
    vU = reduce(vcat, mGP.U)
    Kx = kernelmatrix(ker, [x], Z)

    m = Kx*(Kchol \ vU)
    
    Kxx = kernelmatrix(ker, [x], [x]) # noise term might also be needed here? thoughts!
    std = sqrt.(diag(Kxx - Kx * (Kchol \ Kx')))
    s = randn(length(x)) .* std
    
    fx = (m .+ s)
    
    m = length(mGP.baseZ)
    # println(x)
    if length(Z) < m + mGP.memsize
        mGP.U = vcat(mGP.U, [fx])
        mGP.Z = vcat(mGP.Z, [x])
        
        mGP.Kchol = cholupdate(Kchol, Kx, Kxx + mGP.σ_n*I)        
    else
        l = m + popfirst!(mGP.iter)
        d = length(x)
        mGP.U[l] = fx
        mGP.Z[l] = x
        offset = collect(-d+1:0)
        idx = offset .+ d*l
        Kx[:, idx] = Kxx + (mGP.σ_n + mGP.σ_m)*I
        
        kcholmod!(mGP.Kchol, Kx, l)
        
    end
    
#     K = kernelmatrix(ker, mGP.Z) + mGP.σ_n * I
#     mGP.Kchol = cholesky(K)
    
    # TODO: think about the noise term here
#     d = length(x) #[:, end-d+1:end]
#     Kx2 = kernelmatrix(ker, [x], mGP.baseZ)
#     mGP.Kchol = npODEs.cholupdate(mGP.BaseKchol, Kx2, Kxx + mGP.σ_n*I)

    return fx
end





function computeParts(sgp)
    return _computeParts(sgp, sgp.inP, sgp.method)
end

function _computeParts(sgp, indP::NTuple{2, Array{<:Array{<:Real,1},1}}, method::M) where M
    Z = indP[1]
    U = indP[2]
    σ_n = sgp.σ_n
    ker = sgp.kernel
    # vU = reduce(vcat, U)
    
    K = kernelmatrix(ker, Z) + σ_n * I
    Kchol = cholesky(K)
    return Kchol, U
end

# see Bijl et al, 2015
function _computeParts(sgp, indP::NTuple{3, Array{<:Array{<:Real,1},1}}, method::M) where M
    Z = indP[1]
    X = indP[2]
    Y = indP[3]
    d = length(Y[1])
    vY = reduce(vcat, Y)
    σ_n = sgp.σ_n
    ker = sgp.kernel
    # vU = reduce(vcat, U)
    
    Kff = kernelmatrix(ker, X)
    Kuu = kernelmatrix(ker, Z)
    Kfu = kernelmatrix(ker, X, Z)
    Qff = Symmetric(Kfu * ( Kuu \ Kfu' ))
    Λ = Diagonal(diag( Kff - Qff) .+ σ_n)

    Σtilde = Kuu / GPODEs._computesigma(Kuu, Kfu, Λ)
    # Σtilde = Kuu * inv(GPODEs._computesigma(Kuu, Kfu, Λ))

    vU = (Σtilde)  * (Kfu' * (Λ \ vY))
    U = [u[:] for u in eachcol(reshape(vU, d, :))]

    # σ_a = 1e-1
    Kchol = cholesky(Kuu + σ_n*I)
    return Kchol, U
end