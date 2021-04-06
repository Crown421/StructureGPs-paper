function cholupdate(chol, Bt, C)
    d = size(C, 1)
    Lhat = zeros(size(chol).+size(C))

    Bttilde = Bt/chol.U
    Ctilde = C - Bttilde * Bttilde'
    Ltilde = cholesky(Ctilde).L

    Lhat[1:size(chol,1), 1:size(chol,2)] = chol.L
    # do better with lower triangular?
    Lhat[end-size(Bttilde,1)+1:end, 1:size(Bttilde,2)] = Bttilde
    # Lhat[1:size(Bttilde,2), end-size(Bttilde,1)+1:end] = permutedims(Bttilde)
    Lhat[end-size(Ltilde,1)+1:end, end-size(Ltilde,1)+1:end] = Ltilde
    # Lhat = hcat(vcat(chol.L, Bttilde), vcat(zeros(size(Bttilde,2), size(Bttilde,1)), Ltilde))
    cholhat = Cholesky(Lhat, :L, 0)
end

# # TODO: combine Code, also improve, to change chol.factors (should make in place much easier)
# function cholupdate!(chol, Bt, C)
#     d = size(C, 1)
#     Lhat = zeros(size(chol).+size(C))

#     Bttilde = Bt/chol.U
#     Ctilde = C - Bttilde * Bttilde'
#     Ltilde = cholesky(Ctilde ).L

#     Lhat[1:size(chol,1), 1:size(chol,2)] = chol.L
#     # do better with lower triangular?
#     Lhat[end-size(Bttilde,1)+1:end, 1:size(Bttilde,2)] = Bttilde
#     # Lhat[1:size(Bttilde,2), end-size(Bttilde,1)+1:end] = permutedims(Bttilde)
#     Lhat[end-size(Ltilde,1)+1:end, end-size(Ltilde,1)+1:end] = Ltilde
#     # Lhat = hcat(vcat(chol.L, Bttilde), vcat(zeros(size(Bttilde,2), size(Bttilde,1)), Ltilde))
#     chol.factors = Lhat
# end

function cholmod!(chol, c, k)
    # ToDo dimension check?
    L11 = chol.L[1:k-1, 1:k-1]
    l22 = chol.L[k, k]
    l32 = chol.L[k+1:end, k]
    L31 = chol.L[k+1:end, 1:k-1]
    L33 = chol.L[k+1:end, k+1:end]
    
    c12 = c[1:k-1]
    c22 = c[k]
    c32 = c[k+1:end]

    lb12 = L11 \ c12
    lb22 = sqrt(c22 - lb12'*lb12)
    lb32 = (c32 - L31*lb12)/lb22 #./ 1.0136849372920411

    w2 = deepcopy(lb32)
    w1 = deepcopy(l32)
    
    L33ch = Cholesky(deepcopy(L33), :L, 0)
    lowrankupdate!(L33ch, w1)
    lowrankdowndate!(L33ch, w2)
    
    if chol.uplo == 'L'
        chol.factors[k, 1:k-1] = lb12
        chol.factors[k,k] = lb22
        chol.factors[k+1:end,k] = lb32
        chol.factors[k+1:end, k+1:end] = L33ch.L  
    else
        chol.factors[1:k-1, k] = lb12'
        chol.factors[k,k] = lb22'
        chol.factors[k, k+1:end] = lb32'
        chol.factors[k+1:end, k+1:end] = L33ch.L' 
    end
    chol
end

cholmod(chol, c, k) = cholmod!(deepcopy(chol), c, k)

function kcholmod!(Kchol, Kx, l)
    d = size(Kx, 1)
    offset = collect(-d+1:0)
    idx = offset .+ d*l

    for i in 1:d
    # for i in d:-1:1
        cholmod!(Kchol, Kx[i, :], idx[i])
    end
end