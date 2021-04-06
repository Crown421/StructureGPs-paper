using GaussianProcessODEs
using gpIntegration
gpI = gpIntegration
using Test
using LinearAlgebra
using KernelFunctions

@testset "cholmod" begin
    n = 30
    A = rand(n,n)
    A = A*A'
    Ach = cholesky(A)

    k = rand(1:n)
    # k = 4
    c = rand(n)
    c = c ./(4*norm(c))
    c[k] = 1

    Ah = deepcopy(A)
    Ah[k, :] = c
    Ah[:, k] = c

    if !isposdef(Ah)
        println("Randomly generated a bad c (modified A not pos def), test will fail")
    end

    Bch = deepcopy(Ach)
    # Bch = Cholesky(Ach.L, :L, 0)
    Bch2 = gpI.cholmod(Bch, c, k)
    gpI.cholmod!(Bch, c, k)

    Test.@test !(A ≈ Ah)
    Test.@test cholesky(Ah).L ≈ Bch.L
    Test.@test cholesky(Ah).L ≈ Bch2.L

    Test.@test Bch.L*Bch.L' ≈ Ah
    Test.@test Bch2.L*Bch2.L' ≈ Ah 

end

@testset "kcholmod" begin

    d = 2
    n = 100
    ker = psmkernel(ones(d+1).*0.5);
    Zt = [rand(d).*10 for i in 1:n ]
    Kt = kernelmatrix(ker, Zt)
    Ktchol = cholesky(Kt)
    
    l = rand(1:n)
    c = rand(2)
    Ztmod = deepcopy(Zt)
    Ztmod[l] = c
    Ktmod = kernelmatrix(ker, Ztmod)
    Ktmodchol = cholesky(Ktmod)
    Kx = kernelmatrix(ker, [c], Ztmod)

    gpI.kcholmod!(Ktchol, Kx, l)
    Test.@test Ktchol.L ≈ Ktmodchol.L

end
