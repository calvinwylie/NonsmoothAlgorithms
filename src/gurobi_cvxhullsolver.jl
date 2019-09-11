using Gurobi

struct GurobiConvexHullModel
    gmodel :: Gurobi.Model
    # G :: Array{Vector{Float64}, 1}
    GtG :: Matrix{Float64}
    function GurobiConvexHullModel(k)
        GtG = zeros(Float64, k, k)
        gmodel = (gurobi_model( Gurobi.Env(),
                                H = zeros(Float64, k, k),
                                f = zeros(Float64, k),
                                Aeq = ones(Float64, 1,k),
                                beq = [1.0],
                                lb = 0.0 ))
        setparam!(gmodel, "OutputFlag", 0)
        setparam!(gmodel, "Method", 0)
        setparam!(gmodel, "FeasibilityTol", 1e-9)
        setparam!(gmodel, "OptimalityTol", 1e-9)
        setparam!(gmodel, "BarConvTol", 1e-14)
        setparam!(gmodel, "BarIterLimit", 10000)
        # setparam!(gmodel, "BarCorrectors", 10)
        # setparam!(g_model, "BarHomogeneous", 1)
        # setparam!(g_model, "Quad", 1)
        # setparam!(g_model, "ObjScale", -0.5)
        # setparam!(g_model, "MarkowitzTol", 0.999)
        # setparam!(g_model, "Crossover", 4)
        # setparam!(g_model, "NumericFocus", 0)
        return new(gmodel, GtG)
    end
end


# Projects b onto the convex hull of the vectors G
function solve!(model :: GurobiConvexHullModel, G :: Matrix{Float64}, b)
    add_qpterms!(model.gmodel, G'*G - model.GtG)
    if !ismissing(b)
        set_objcoeffs!(model.gmodel, G'*b)
    end
    update_model!(model.gmodel)
    # model.G = G
    model.GtG[:,:] = G'*G
    optimize(model.gmodel)
    # println(get_status(model.gmodel))
    return get_objval(model.gmodel), get_solution(model.gmodel)
end

function solve!(model :: GurobiConvexHullModel, G :: Array{Vector{Float64}, 1})
    G = hcat(G...)
    return solve!(model, G'*G)
end