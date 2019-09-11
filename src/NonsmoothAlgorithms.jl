module NonsmoothAlgorithms
using NonsmoothFunctions

include("bfgs.jl")
export bfgs_solve

include("bundle_methods.jl")
export multiple_cut_bundle_solve, proximal_cutting_plane_solve, cutting_plane_solve

include("gurobi_cvxhullsolver.jl")
export GurobiConvexHullModel
export solve!

include("bundle_newton.jl")
export bundlenewtonsolve, bundlenewtonfirstordersolve

end