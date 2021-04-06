export radialgrid, rectanglegrid

function rectanglegrid(borders, npoints::Array{Int, 1})
    tmp = [range(borders[i][1], borders[i][end], length = npoints[i]) for i in 1:length(borders)]
    Z = collect.(collect(Iterators.product(tmp...)))[:]
end

function rectanglegrid(borders, nGrid::Int)
    maxstep = maximum(reduce(vcat, diff.(borders)./nGrid))
    npoints = ceil.(reduce(vcat, (diff.(borders)./maxstep)))
    npoints = Int.(max.(npoints, 1.0))

    return rectanglegrid(borders, npoints)
end


function radialgrid(radius, nGridPoints; origin = [0,0])
    r = range(0, 3, length = nGridPoints+1)[2:end]
    th = collect(range(0, 2*pi, length = nGridPoints+1))[1:end-1]
    Z = [ [r*cos(th), r*sin(th)] for th in th, r in r][:] .+ [origin]
end

    