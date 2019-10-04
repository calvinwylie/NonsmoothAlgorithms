using NonsmoothModels

struct NewtonBundle{T <: Real}
    model
    n :: Integer
    k :: Integer
    Y :: Array{T, 2} # Bundle points
    F :: Array{T, 1} # Function values
    G :: Array{T, 2} # Gradients
    H :: Array{T, 3} # Hessians

    U                # Basis for manifold tangent space approximation
    v
    R :: Array{T,3} # Projected Hessians

    function NewtonBundle(model, Y :: Array{Vector{T}, 1}) where {T <: Real}
        k = length(Y)
        n = length(Y[1])
        Y = hcat(Y...)
        F = zeros(T, k)
        G = zeros(T, n, k)
        H = zeros(T, n, n, k)
        for j = 1:k
            # println(Y[:,j],)
            F[j] = evaluate!(model, Y[:,j], view(G,:,j), view(H,:,:,j))
            
            # H[:,:,j] = Matrix{Float64}(I, n, n)
        end

        pivot = true
        @views GG = G[:,1:k-1] .- G[:,k]
        @views b = [dot(G[:,j], Y[:,j]) - F[j] for j = 1:k-1] .- (dot(G[:,k], Y[:,k]) - F[k])
        QR = qr(GG, Val(pivot))
        U = QR.Q[:, k:end]
        V = QR.Q[:,1:k-1]*(QR.R \ (pivot ? QR.P' : I))
        v = V*(GG'*V \ b)

        R = zeros(T, n-k+1, n-k+1, k)
        for j = 1:k
            try
                writereducedhessian!(model, Y[:,j], U, view(R,:,:,j))
            catch MethodError
                if j == 1 
                    @warn "No reduced Hessian method, falling back to explicit computation."
                end
                mul!(view(R,:,:,j), U'*view(H,:,:,j), U)
            end
        end

        return new{T}(model, n, k, Y, F, G, H, U, v, R)
    end
end

gradientmatrix(bundle :: NewtonBundle) = bundle.G
bundlematrix(bundle :: NewtonBundle) = bundle.Y
bundlediameter(bundle :: NewtonBundle) = 
    maximum( norm(bundle.Y[:,i] - bundle.Y[:,j]) for i = 1:bundle.k for j = 1:bundle.k )

function bundleslope(simplexqpsolver :: Function, bundle :: NewtonBundle)
    G = gradientmatrix(bundle)
    _, lambda = simplexqpsolver(G, missing)
    return norm(G*lambda), lambda
end

function convexhulldist(simplexqpsolver :: Function, G, b)
    # G = gradientmatrix(bundle)
    _, lambda = simplexqpsolver(G, b)
    return norm(G*lambda - b), lambda
end

function linearizationerrors(x, bundle :: NewtonBundle, eta)
    f_x = objective(bundle.model, x) + (eta/2)*norm(x)^2
    return [f_x - bundle.F[i] - (eta/2)*norm(bundle.Y[:,i])^2 - dot(bundle.G[:,i] + eta*bundle.Y[:,i], x - bundle.Y[:,i]) for i = 1:bundle.k]
end

function updatereducedfactorization!(bundle :: NewtonBundle)
    n = bundle.n
    k = bundle.k
    Y = bundle.Y
    F = bundle.F
    G = bundle.G
    H = bundle.H

    pivot = true
    @views GG = G[:,1:k-1] .- G[:,k]
    @views b = [dot(G[:,j], Y[:,j]) - F[j] for j = 1:k-1] .- (dot(G[:,k], Y[:,k]) - F[k])
    QR = qr(GG, Val(pivot))
    U = QR.Q[:, k:end]
    V = QR.Q[:,1:k-1]*(QR.R \ (pivot ? QR.P' : I))
    v = V*(GG'*V \ b)

    bundle.U .= U 
    bundle.v .= v
    for j = 1:k
        try
            writereducedhessian!(bundle.model, Y[:,j], bundle.U, view(bundle.R,:,:,j))
        catch MethodError
            mul!(view(bundle.R,:,:,j), U'*view(H,:,:,j), U)
        end
        # println(norm(bundle.R[:,:,j] - bundle.R[:,:,j]', 1))
    end

    return bundle
end

function updatebundle!(bundle :: NewtonBundle, r, y)
    k = bundle.k
    Y = bundle.Y
    F = bundle.F
    G = bundle.G
    H = bundle.H

    Y[:,r] = y
    F[r] = evaluate!(bundle.model, y, view(G,:,r), view(H,:,:,r))

    updatereducedfactorization!(bundle :: NewtonBundle)

    return bundle
end

function bfgsupdatebundle!(bundle :: NewtonBundle, r, x)
    k = bundle.k
    Y = bundle.Y
    F = bundle.F
    G = bundle.G
    H = bundle.H

    s = x - Y[:,r]

    Y[:,r] = x
    g_prev = G[:,r]
    F[r] = evaluate!(bundle.model, x, view(G,:,r))
    y = G[:,r] - g_prev

    # BFGS
    Hrs = H[:,:,r]*s
    H[:,:,r] = H[:,:,r] + (y*y')/dot(s,y) - (Hrs*s'*H[:,:,r]')/dot(s, Hrs)

    # SR-1
    # H[:,:,r] = H[:,:,r] + (1/dot(y - H[:,:,r]*s, s))*(y - H[:,:,r]*s)*(y - H[:,:,r]*s)'


    updatereducedfactorization!(bundle :: NewtonBundle)

    return bundle
end

function directsolve(bundle :: NewtonBundle, lambda, eta)
    n = bundle.n
    k = bundle.k
    Y = bundle.Y
    F = bundle.F
    G = bundle.G
    H = bundle.H

    Z = zeros(n,k)
    for j = 1:k
        Z[:,j] = Y[:,j] 
        # Z[:,j] = Y[:,j] - Y*lambda
    end

    eta = maximum([eigmax(-H[:,:,i]) for i = 1:k])
    # eta = minimum(eig)
    # println(eigvals(H[:,:,1]))

    A = zeros(n+1+k, n+1+k)
    A[1:n,1:n] = sum(lambda[j]*H[:,:,j] for j=1:k)
    A[1:n, n+2:n+1+k] = G + eta*Z
    A[n+2:n+1+k, 1:n] = (G + eta*Z)'
    A[n+1, n+2:n+1+k] = ones(1,k)
    A[n+2:n+1+k, n+1] = ones(k,1)
    # A = Symmetric(A)

    b = zeros(n+1+k)
    # b[1:n] = sum((lambda[j]*H[:,:,j]*Y[:,j] for j=1:k)) + eta*Y*lambda
    b[1:n] = sum((lambda[j]*H[:,:,j]*Y[:,j] for j=1:k)) - G*lambda
    # b[n+1] = 1
    b[n+1] = 0
    for j = 1:k
        # b[n+1+j] = -F[j] + (eta/2)*norm(Y[:,j])^2 + dot(G[:,j], Y[:,j])
        b[n+1+j] = -F[j] - (eta/2)*norm(Z[:,j])^2 + dot(G[:,j] + eta*Z[:,j], Y[:,j])
    end

    # return (A \ b)[1:n]

    S = bunchkaufman(Symmetric((1/2)*(A + A')))
    sol = S \ b
    return sol[1:n]
end

function reducednewtonsolve(bundle :: NewtonBundle, lambda)
    k = bundle.k
    Y = bundle.Y
    G = bundle.G
    R = bundle.R
    U = bundle.U
    v = bundle.v

    L = (sum(lambda[j]*R[:,:,j] for j = 1:k))

    xu = L \ sum(lambda[j]*( R[:,:,j]*U'*(Y[:,j] - v) - U'*G[:,j]) for j=1:k)
    
    return U*xu + v

    # du = L \ -U'*G*lambda
    # return U*(U'*(Y*lambda - v) + du) + v
end

function reducednewtonlinesearchsolve(bundle :: NewtonBundle, lambda, t)
    k = bundle.k
    Y = bundle.Y
    G = bundle.G
    R = bundle.R
    U = bundle.U
    v = bundle.v

    L = (sum(lambda[j]*t*R[:,:,j] for j = 1:k))

    xu = L \ sum(lambda[j]*(t*R[:,:,j]*U'*(Y[:,j] - v) - U'*G[:,j]) for j=1:k)
    # xu = L \ sum(lambda[j]*( vec(reducedhessianvecprod(bundle.model, Y[:,j], U, Y[:,j] - v)) - U'*G[:,j]) for j=1:k)
    return U*xu + v

    # du = L \ -U'*G*lambda
    # return U*(U'*(Y*lambda - v) + du) + v
end

function firstorderdirectsolve(bundle :: NewtonBundle, lambda, t, eta)
    n = bundle.n
    k = bundle.k
    Y = bundle.Y
    F = bundle.F
    G = bundle.G
    H = bundle.H

    Z = zeros(n,k)
    for j = 1:k
        Z[:,j] = Y[:,j] 
        # Z[:,j] = Y[:,j] - Y*lambda
        # Z[:,j] = Y[:,j] - prev_x
    end

    # eta = 2*maximum([eigmax(-H[:,:,i]) for i = 1:k])

    A = zeros(n+1+k, n+1+k)
    A[1:n,1:n] = t*Matrix{Float64}(I, n, n)
    A[1:n, n+2:n+1+k] = G + eta*Z
    A[n+2:n+1+k, 1:n] = (G + eta*Z)'
    A[n+1, n+2:n+1+k] = ones(1,k)
    A[n+2:n+1+k, n+1] = ones(k,1)

    b = zeros(n+1+k)
    # b[1:n] = sum((lambda[j]*H[:,:,j]*Y[:,j] for j=1:k)) + eta*Y*lambda
    b[1:n] = t*Y*lambda - G*lambda
    # b[n+1] = 1
    b[n+1] = 0
    for j = 1:k
        # b[n+1+j] = -F[j] + (eta/2)*norm(Y[:,j])^2 + dot(G[:,j], Y[:,j])
        b[n+1+j] = -F[j] - (eta/2)*norm(Z[:,j])^2 + dot(G[:,j] + eta*Z[:,j], Y[:,j])
    end

    # return (A \ b)[1:n]

    S = bunchkaufman(Symmetric((1/2)*(A + A')))
    sol = S \ b
    return sol[1:n]
end

function upperbound(bundle :: NewtonBundle)
    return minimum(bundle.F)
end

function lowerbound(bundle :: NewtonBundle, lambda, rho)
    G = gradientmatrix(bundle)
    Y = bundlematrix(bundle)
    F = bundle.F
    k = bundle.k

    y_bar = Y*lambda .- (rho > 0 ? (1/rho)*G*lambda : 0)
    bound = 0
    for j = 1:k
        z = y_bar - Y[:,j]
        bound += lambda[j]*(F[j] + dot(G[:,j], z) + (rho > 0 ? (rho/2)*norm(z)^2 : 0) )
    end
    return bound
end

function bundlenewtonsolve(model, simplexqpsolver, Y, eta, rho, f_tol, max_iter)
    k = length(Y)
    n = length(Y[1])

    X = zeros(n, max_iter)
    upper_bound = zeros(max_iter)
    lower_bound = zeros(max_iter)
    diameter_hist = zeros(max_iter)
    slope_hist = zeros(max_iter)

    bundle = NewtonBundle(model, Y)
    slope, lambda = bundleslope(simplexqpsolver, bundle)

    mu = lambda

    println("Diameter\tSlope\tUpper bound\tLower bound")
    final_iter = max_iter
    for i = 1:max_iter

        lower_bound[i] = lowerbound(bundle, lambda, rho)
        upper_bound[i] = i > 1 ? min(upperbound(bundle), upper_bound[i-1]) : upperbound(bundle)
        diameter_hist[i] = bundlediameter(bundle)
        slope_hist[i] = slope
        println(i, "\t", diameter_hist[i], "\t", slope_hist[i], "\t", upper_bound[i], "\t", lower_bound[i])

        x = try
            # reducednewtonsolve(bundle, lambda)
            directsolve(bundle, lambda, eta)
        catch
            println("Error, stopping...")
            final_iter = i-1
            break
        end

        X[:,i] = x

        # Unsafe, find a better way to do this
        d = zeros(bundle.k)
        G = gradientmatrix(bundle)
        # g_x = gradient(model, x)
        g_x = similar(x)
        evaluate!(model, x, g_x)

        if ismissing(g_x)
            println("Nonsmooth point encountered, stopping...")
            final_iter = i-1
            break
        end

        weights = []
        for j = 1:bundle.k
            g_j = G[:,j]
            G[:,j] = g_x
            try
                dist, w = bundleslope(simplexqpsolver, bundle)
                d[j] = dist
                push!(weights, w)
            catch e
                @warn e
                d[j] = Inf
                push!(weights, missing)
            end
            G[:,j] = g_j
        end
        r = argmin(d)
        updatebundle!(bundle, r, x)
        # bfgsupdatebundle!(bundle, r, x)

        lambda = weights[r]
        slope = d[r]

        if false #upper_bound[i] - lower_bound[i] < f_tol
            println("Stopping: upper bound - lower bound <", f_tol)
            final_iter = i
            break
        end
    end

    return bundle.Y, X[:, 1:final_iter], upper_bound[1:final_iter], lower_bound[1:final_iter], diameter_hist[1:final_iter], slope_hist[1:final_iter]
end

function bundlenewtonfirstordersolve(model, simplexqpsolver, Y, eta, rho, f_tol, max_iter)
    k = length(Y)
    n = length(Y[1])

    X = zeros(n, max_iter)
    upper_bound = zeros(max_iter)
    lower_bound = zeros(max_iter)

    bundle = NewtonBundle(model, Y)
    slope, lambda = bundleslope(simplexqpsolver, bundle)

    final_iter = max_iter
    for i = 1:max_iter

        lower_bound[i] = lowerbound(bundle, lambda, rho)
        upper_bound[i] = i > 1 ? min(upperbound(bundle), upper_bound[i-1]) : upperbound(bundle)

        if i == 1
            println("Diameter\tSlope\tUpper bound\tLower bound")
        end
        println(bundlediameter(bundle), "\t", slope, "\t", upper_bound[i], "\t", lower_bound[i])

        accept = false
        t = 0.25
        alpha = 0
        beta = Inf
        numtrials = 0
        while !accept
            x = firstorderdirectsolve(bundle, lambda, t, eta)
            # x = reducednewtonlinesearchsolve(bundle, lambda, t)
            # g_x = gradient(model, x)
            g_x = similar(x)
            evaluate!(model, x, g_x)
            d = zeros(bundle.k)
            G = gradientmatrix(bundle)
            weights = []
            for j = 1:bundle.k
                g_j = G[:,j]
                G[:,j] = g_x
                try
                    dist, w = bundleslope(simplexqpsolver, bundle)
                    d[j] = dist
                    push!(weights, w)
                catch e
                    @warn e
                    d[j] = Inf
                    push!(weights, missing)
                end
                G[:,j] = g_j
            end
            r = argmin(d)

            # newdiameter = maximum( norm(bundle.Y[:,i] - bundle.Y[:,j]) for i = 1:bundle.k if i != r for j = 1:bundle.k if j != r)
            # newdiameter = max(newdiameter, maximum( norm(x - bundle.Y[:,i]) for i = 1:bundle.k if i != r ))

            # if bundle.F[r] - objective(model, x) > 0
            # if bundle.F[r] - objective(model, x) >= 1e-5*(-dot(bundle.G[:,r], x - bundle.Y[:,r]))
            # if objective(model, x) <= bundle.F[r] + dot(bundle.G[:,r], x - bundle.Y[:,r]) + (t/2)*norm(x - bundle.Y[:,r])^2
            # if d[r] < slope || newdiameter < bundlediameter(bundle)
            # if bundle.F[argmax(bundle.F)] - objective(model, x) >= 1e-5*(-dot(bundle.G[:,argmax(bundle.F)], x - bundle.Y[:,argmax(bundle.F)]))
            # if objective(model, x) < bundle.F[r] - (1e-5/t)*norm(x - bundle.Y[:,r])^2
            if true
                accept = true
                println("\t", t, "\t", numtrials)
                X[:,i] = x
                updatebundle!(bundle, r, x)
                lambda = weights[r]
                slope = d[r]
            elseif numtrials > 30
                break
            else
                t = t*2
                numtrials += 1
            end
        end
        if !accept 
            println("Stopping: Could not find decrease")
            final_iter = i
            break
        end
        # println(l_bound, "\t", u_bound)

        if false #upper_bound[i] - lower_bound[i] < f_tol
            println("Stopping: upper bound - lower bound <", f_tol)
            final_iter = i
            break
        end
    end

    return Y, X[:, 1:final_iter], upper_bound[1:final_iter], lower_bound[1:final_iter]
end


