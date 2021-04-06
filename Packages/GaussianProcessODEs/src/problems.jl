# export ckepler, ckepler!, pkepler, pkepler!
export vdp, vdp!, basic!, basic


m = 1;
k = 1.0168951928e3;
kh = k/m;

function ckepler!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -kh/(norm([x[1], x[3]])^3) * x[1]
    dx[3] = x[4]
    dx[4] = -kh/(norm([x[1], x[3]])^3) * x[3]
end
function ckepler(x, p, t)
    dx = similar(x)
    ckepler!(dx, x, p, t)
    return dx
end

function pkepler!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = x[1]*x[4]^2 - kh/(x[1]^2)
    dx[3] = x[4]
    dx[4] = -2*x[2]*x[4]/x[1]
end
function pkepler(x, p, t)
    dx = similar(x)
    pkepler!(dx, x, p, t)
    return dx
end


function vdp!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = (1-x[1]^2)*x[2] - x[1]
end
function vdp(x, p, t)
    dx = similar(x)
    vdp!(dx, x, p, t)
    return dx
end


function basic!(dx, x, p, t)
    A = [0. -0.95; 0.95 0.]
    dx[:] = A * x
end

function basic(x, p, t)
    dx = similar(x)
    basic!(dx, x, p, t)
    return dx
end
