using DifferentialEquations
using Flux, DiffEqFlux
using GaussianProcessODEs
using LinearAlgebra

m = 1;
k = 1.0168951928e3;
kh = k/m;
function ckepler!(dx, x, p, t)
    dx[1] = x[3]
    dx[2] = x[4]
    dx[3] = -p/(norm([x[1], x[2]])^3) * x[1] #x[4]
    dx[4] = -p/(norm([x[1], x[2]])^3) * x[2]
end

function ckepler(x, p, t)
    dx = similar(x)
    ckepler!(dx, x, p, t)
    return dx
end;

function pdot(dp,p,q,params,t)
    dp[:] = params(q)
end
function qdot(dq,p,q,params,t) 
    dq[:] = p
end


dudt2 = FastChain((x, p) -> x,
                  FastDense(4, 100, tanh),
                  FastDense(100, 4))