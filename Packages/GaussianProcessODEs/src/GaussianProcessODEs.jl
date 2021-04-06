module GaussianProcessODEs

using DifferentialEquations
using Zygote
using Distances
using VectorizedRoutines: pairwise
using LinearAlgebra
using Optim
using QuadGK
using KernelFunctions
using Measurements 
using ThreadsX

include("kernelfuns.jl")
include("kernelutils.jl")
include("gp_de.jl")
include("kernel_opt.jl")

include("base.jl")
include("problems.jl")
include("utils.jl")
include("hamiltonian.jl")

end # module