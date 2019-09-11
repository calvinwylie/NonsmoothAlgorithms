using JuMP

function multiple_cut_bundle_solve(model, jump_solver, x0, rho, sigma, dual_tol, f_tol, max_iter, callback)
    n = length(x0)
    bundle = [x0]
    duals = []

    center = copy(x0)

    x = copy(x0)
    g_x = similar(x)

    f_x = evaluate!(model, x, g_x)
    f_c = f_x

    rho_min = 1e-6*rho
    rho_max = 1e6*rho

    qpmodel = Model(jump_solver)
    @variable(qpmodel, var_t)
    @variable(qpmodel, var_x[1:n])
    @objective(qpmodel, Min, var_t + (rho/2)*sum( (var_x[i] - center[i])^2 for i = 1:n))
    @constraint(qpmodel, var_t >= f_x + dot(g_x, var_x - x))

    for i = 1:max_iter
        optimize!(qpmodel)
        # println(termination_status(qpmodel))

        x = value.(var_x)
        t = value(var_t)
        f_x = evaluate!(model, x, g_x)
        if !ismissing(callback) 
            callback(x, t)
        end

        constraints = all_constraints(qpmodel, GenericAffExpr{Float64,VariableRef}, MOI.GreaterThan{Float64})
        mu = dual.(constraints)
        for j = 1:length(mu)
            if mu[j] < dual_tol
                delete(qpmodel, constraints[j])
            end
        end
        deleteat!(bundle, findall(mu .< dual_tol))

        if f_c - t < 0
            println("Stopping: numerical error (gap = ", f_c - t, " < 0)")
            break
        elseif f_c - t <= f_tol
            println("Stopping: gap = ", f_c - t)
            break
        else
            if f_c - f_x >= sigma*(f_c - t)
                center .= x
                f_c = f_x
                @objective(qpmodel, Min, var_t + (rho/2)*sum( (var_x[i] - center[i])^2 for i = 1:n))
                rho = max(rho_min, rho / 2)
            else
                rho = min(rho_max, rho*2)
            end
            @constraint(qpmodel, var_t >= f_x + dot(g_x, var_x - x))
            push!(bundle, x)
        end
    end 

    return bundle
end

function proximal_cutting_plane_solve(model, jump_solver, x0, rho, f_tol, max_iter, callback)
    n = size(x0,1)
    Y = hcat(x0)
    duals = []

    x = copy(x0)
    g_x = similar(x)
    f_x = evaluate!(model, x, g_x)

    lpmodel = Model(jump_solver)
    @variable(lpmodel, var_t)
    @variable(lpmodel, var_x[1:n])
    @objective(lpmodel, Min, var_t + (rho/2)*sum( (var_x[i] - x[i])^2 for i = 1:n))
    @constraint(lpmodel, var_t >= f_x + dot(g_x, var_x - x))

    for i = 1:max_iter
        optimize!(lpmodel)
        # print(termination_status(lpmodel), "\t")
        x = value.(var_x)
        t = value(var_t)
        f_x = evaluate!(model, x, g_x)
        if !ismissing(callback) 
            callback(x, t)
        end

        if f_x - t < 0
            println("Stopping: numerical error (gap = ", f_x - t, " < 0)")
            break
        elseif f_x - t <= f_tol
            println("Stopping: gap = ", f_x - t)
            break
        else
            @constraint(lpmodel, var_t >= f_x + dot(g_x, var_x - x))
            @objective(lpmodel, Min, var_t + (rho/2)*sum( (var_x[i] - x[i])^2 for i = 1:n))
            Y = hcat(Y, x)
        end
    end 
end

function cutting_plane_solve(model, jump_solver, x0, lb, ub, f_tol, max_iter, callback)
    n = size(x0,1)
    Y = hcat(x0)
    duals = []

    x = copy(x0)
    g_x = similar(x)
    f_x = evaluate!(model, x, g_x)

    lpmodel = Model(jump_solver)
    @variable(lpmodel, var_t)
    @variable(lpmodel, var_x[1:n])
    @objective(lpmodel, Min, var_t)
    @constraint(lpmodel, var_t >= f_x + dot(g_x, var_x - x))
    @constraint(lpmodel, var_x .>= lb)
    @constraint(lpmodel, var_x .<= ub)

    for i = 1:max_iter
        optimize!(lpmodel)
        # print(termination_status(lpmodel), "\t")
        x = value.(var_x)
        t = value(var_t)
        f_x = evaluate!(model, x, g_x)
        if !ismissing(callback) 
            callback(x, t)
        end

        if f_x - t < 0
            println("Stopping: numerical error (gap = ", f_x - t, " < 0)")
            break
        elseif f_x - t <= f_tol
            println("Stopping: gap = ", f_x - t)
            break
        else
            @constraint(lpmodel, var_t >= f_x + dot(g_x, var_x - x))
            Y = hcat(Y, x)
        end
    end 
end
