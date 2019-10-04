using LinearAlgebra

function bfgs_solve(model, x0, c1, c2, max_iters, callback=missing)
    x = x0
    g = similar(x)
    gp = similar(g)
    H = Matrix{eltype(x0)}(I, length(x0), length(x0))

    f = evaluate!(model, x, g)

    for k = 1:max_iters
        p = -H*g
        s = dot(g, p)

        fp = nothing
        t, n_trials = weakwolfe_linesearch(s, c1, c2, 50) do t
            fp = evaluate!(model, x + t*p, gp)
            return fp - f, dot(gp, p)
        end

        if ismissing(t)
            println("BFGS breakdown: linesearch failed")
            break
        end

        x = x + t*p

        H = invhessianupdate(H, t, p, gp - g)   
        if dot(gp, H*gp) < 0
            println("BFGS breakdown: negative curvature")
            break
        end

        if !ismissing(callback) 
            callback(x, H, n_trials)
        end
        f = fp
        g .= gp
    end

    return x
end

# A bisection method for the weak Wolfe conditions
function weakwolfe_linesearch(oracle :: Function, s, c1, c2, max_trials)
    alpha = 0
    beta = Inf
    t = 1
    accept = false
    n_trials = 0
    while !accept
        n_trials += 1

        h, dh = oracle(t)

        if h >= c1*s*t
            beta = t
        elseif dh <= c2*s
            alpha = t
        else
            accept = true
        end

        if !accept
            if beta < Inf
                t = (alpha + beta)/2
            else
                t = 2*alpha
            end
        end

        if n_trials > max_trials && !accept
            return missing, n_trials
        end
    end

    return t, n_trials
end

function invhessianupdate(H, t, p, y)
    V = I - (1/dot(p,y))*p*y'
    return V*H*V' + t*(1/dot(p,y))p*p'
end
