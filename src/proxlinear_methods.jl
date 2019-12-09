# TODO: incorporate the linesearch into the algorithms
function proxgradientlinesearch!(z, smooth_model, ns_model, x, g, t0)
    t = t0
    while true
        prox!(z, ns_model, x - t*g, t)
        if objective(smooth_model, z) <= objective(smooth_model, x) + dot(g, z - x) + (1/(2*t))*norm(z - x)^2
            return t
        else
            t = t/2
        end

        if t < 1/(2^10)
            @error("Linesearch failed")
        end
    end
end

function proxgradientsolve(smooth_model, ns_model, x0, t, GTOL, MAX_ITER, callback=missing)
    n = length(x0)
    x = x0
    z_tmp = similar(x) # Working space for computing gradients and prox operators
    g_tmp = similar(x)

    for k = 1:MAX_ITER
        writegradient!(g_tmp, smooth_model, x)
        prox!(z_tmp, ns_model, x - t*g_tmp, t)
        # t = proxgradientlinesearch!(z_tmp, smooth_model, ns_model, x, g_tmp, t)
        # t = t*2


        g = (1/t)*(x - z_tmp) # "Gradient" map
        
        if norm(g) <= GTOL
            println("Stopping after ", k, " iterations: |g| = ", norm(g))
            return x
        else
            if !ismissing(callback)
                callback(x, g)
            end
            x = x - t*g
        end
    end

    println("Hit iteration limit: ", MAX_ITER)
    return x
end

function fastproxgradientsolve(smooth_model, ns_model, x0, t, GTOL, MAX_ITER, callback=missing)
    n = length(x0)
    x = copy(x0)
    g_tmp = similar(x) # Working space for computing gradients and prox operators
    z_tmp = similar(x)
    x_prev = copy(x)
    y = copy(x)

    for k = 1:MAX_ITER
        y = x + ((k-2)/(k+1))*(x - x_prev)

        writegradient!(g_tmp, smooth_model, y)
        prox!(z_tmp, ns_model, y - t*g_tmp, t)
        # t = proxgradientlinesearch!(z_tmp, smooth_model, ns_model, y, g_tmp, t)
        # t = t*2
        g = (1/t)*(y - z_tmp) # "Gradient" map
        
        if norm(g) <= GTOL
            println("Stopping after ", k, " iterations: |g| = ", norm(g))
            return x
        else
            if !ismissing(callback)
                callback(x, g)
            end
            x_prev = x
            x = y - t*g
        end
    end

    println("Hit iteration limit: ", MAX_ITER)
    return x
end

function secondorderproxgradientsolve(smooth_model, ns_model, x0, t, GTOL, MAX_ITER, callback=missing)
    n = length(x0)
    x = x0
    tmp = similar(x) # Working space for computing gradients and prox operators
    w = similar(x)
    H = zeros(n,n)

    f = x -> objective(smooth_model, x) + objective(ns_model, x)

    writegradient!(tmp, smooth_model, x)
    prox!(tmp, ns_model, x - t*tmp, t)
    g = (1/t)*(x - tmp) # "Gradient" map
    # xg = x - t*g
    Tx = x - t*g
    # v = (1/t)*(x - xg) - H*(x - xg)
    v = (1/t)*(x - Tx)
    writegradient!(tmp, smooth_model, x); v = v - tmp
    writegradient!(tmp, smooth_model, Tx); v = v + tmp
    x = Tx

    for k = 1:MAX_ITER
        # If some condition holds try a linearization step
        # if k > 0
        if sum(x .!= 0) < 40
            writehessian!(H, smooth_model, x)
            subgradientderivativesolve!(w, ns_model, x, H, -v)
            f_accept = f(x)

            accept = false
            r = 1
            while !accept
                if r < 1/(2^5)
                    break
                end
                if f(x + r*w) < f_accept
                    accept = true
                else
                    r = r/2
                end
            end
            if accept
                x = x + r*w
            end
        end

        # Regular prox-gradient step
        writegradient!(tmp, smooth_model, x)
        prox!(tmp, ns_model, x - t*tmp, t)
        g = (1/t)*(x - tmp) # "Gradient" map
        Tx = x - t*g

        # Restore to the graph
        v = (1/t)*(x - Tx)
        writegradient!(tmp, smooth_model, x); v = v - tmp
        writegradient!(tmp, smooth_model, Tx); v = v + tmp

        x = Tx

        if !ismissing(callback)
            callback(x, g)
        end

        if norm(g) <= GTOL
            println("Stopping after ", k, " iterations: |g| = ", norm(g))
            return x
        end
    end

    println("Hit iteration limit: ", MAX_ITER)
    return x
end

function fastsecondorderproxgradientsolve(smooth_model, ns_model, x0, t, GTOL, MAX_ITER, callback=missing)
    n = length(x0)
    x = x0
    g_tmp = similar(x) # Working space for computing gradients and prox operators
    z_tmp = similar(x)
    v_tmp = similar(x)
    w = similar(x)
    H = zeros(n,n)

    xprev = x

    f = x -> objective(smooth_model, x) + objective(ns_model, x)

    writegradient!(g_tmp, smooth_model, x)
    prox!(z_tmp, ns_model, x - t*g_tmp, t)
    # t = proxgradientlinesearch!(z_tmp, smooth_model, ns_model, x, g_tmp, t)
    # t = t*2
    g = (1/t)*(x - z_tmp) # "Gradient" map
    # xg = x - t*g
    Tx = x - t*g
    # v = (1/t)*(x - xg) - H*(x - xg)
    v = (1/t)*(x - Tx)
    writegradient!(v_tmp, smooth_model, x); v = v - v_tmp
    writegradient!(v_tmp, smooth_model, Tx); v = v + v_tmp
    xprev = x; x = Tx

    for k = 1:MAX_ITER
        # If some condition holds try a linearization step
        if sum(x .!= 0)^3 <= size(smooth_model.AtA,1)*size(smooth_model.AtA,2) #k > 300
        # if k > 100
        # if sum(x .!= 0) < 50
            # println(k)
            writehessian!(H, smooth_model, x)
            subgradientderivativesolve!(w, ns_model, x, H, -v)
            f_accept = f(x)

            accept = false
            r = 1
            while !accept
                if r < 1/(2^8)
                    # r = 1
                    break
                end
                if f(x + r*w) < f_accept
                    accept = true
                else
                    r = r/2
                end
            end
            if accept
                y = x + r*w
            else
                y = x + ((k-2)/(k+1))*(x - xprev)
            end
        else
            y = x + ((k-2)/(k+1))*(x - xprev)
        end

        # Regular prox-gradient step
        writegradient!(g_tmp, smooth_model, y)
        prox!(z_tmp, ns_model, y - t*g_tmp, t)
        # t = proxgradientlinesearch!(z_tmp, smooth_model, ns_model, y, g_tmp, t)
        # t = t*2
        g = (1/t)*(y - z_tmp) # "Gradient" map
        Ty = y - t*g

        # Restore to the graph
        v = (1/t)*(y - Ty)
        writegradient!(v_tmp, smooth_model, y); v = v - v_tmp
        writegradient!(v_tmp, smooth_model, Ty); v = v + v_tmp

        xprev = x; x = Ty

        if !ismissing(callback)
            callback(x, g)
        end

        if norm(g) <= GTOL
            println("Stopping after ", k, " iterations: |g| = ", norm(g))
            return x
        end
    end

    println("Hit iteration limit: ", MAX_ITER)
    return x
end

