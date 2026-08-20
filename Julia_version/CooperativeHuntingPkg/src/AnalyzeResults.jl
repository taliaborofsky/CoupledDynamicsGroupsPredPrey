module AnalyzeResults


using UnPack

export get_p, get_meanx, get_prob_in_x
# methods get_p
    function get_p(g::AbstractMatrix, x_max::Int)
        x = 1:x_max
        return sum(x .* g, dims = 1)  # Return a vector
    end

    function get_p(g::AbstractVector, x_max::Int)
        x = 1:x_max
        return sum(x .*g) # returns a scalar
    end
    function get_p(g::AbstractVector)
        x_max = length(g)
        get_p(g,x_max)
    end

    function get_meanx(g::AbstractMatrix, x_max::Int, p::AbstractMatrix)
        """
        Average group size any individual is in when `p` is a vector.
        g is a 2d matrix
        """
        x_vec = 1:x_max
        numerator = @. (x_vec^2) * g
        #mask = (p .> 1e-10) .& all(g .> 0, dims=1)
        mask = p.>1e-10
        numerator_sum = sum(numerator, dims=1)
        ans = copy(p)
        ans[mask] = @. numerator_sum[mask] / p[mask]
        @. ans[!mask] = 1.0
        return ans
    end


# methods get_meanx
    function get_meanx(g::AbstractVector, x_max::Int, p::Number)
        #=
        Average group size any individual is in when `p` is a scalar.
        will add method for p a vector and g a matrix later
        =#
        x_vec = 1:x_max
        numerator = @. x_vec^2 * g
        if p < 1e-10 #&& all(g .< 1e-10)
            return 1.0
        else
            return sum(numerator) / p
        end
    end

    function get_meanx(g, x_max)
    #=
        get_meanx without p given
    =#
        p = get_p(g,x_max)
        mean_x = get_meanx(g, x_max, p)
        return mean_x
    end

# methods get_prob_in_x
    function get_prob_in_x(g::AbstractMatrix, p::AbstractVector, x_max)
    # find prob in group of size x, for g a matrix and p a vector
    # if length of p is n, then dimensions of g need to be n x x_max

        x=1:x_max
        num_in_gx = @. g * x'
        prob_in_x = @. num_in_gx / p
    end

    function get_prob_in_x(g::AbstractMatrix, p::Number, x_max)
        # find prob in group of size x, for g a matrix and p a vector
            x=1:x_max
            num_in_gx = @. g * x'
            prob_in_x = @. num_in_gx / p
        end
        
    function get_prob_in_x(g::AbstractVector, p::Number, x_max)
        x = 1:x_max
        num_in_gx = @. g * x
        prob_in_x = @. num_in_gx / p
    end


# conditions to check if prediction matches mangel clark



    function condition_mc(g₁, x̂, W, params)
        """

        Condition for which the group size experienced by the most individuals is 
        the mangel and clark prediction, x̂
        Inputs: g1 at equilibrium (VECTOR), x̂ (mangel and clark prediction)

        Returns True/False

        NEED TO CHECK
        """
        C = calculate_C(W, params)

        # lower bound
        lower = (x̂ - 1) / (x̂ * C[x̂] * C[x̂ - 1])

        # now find upper bound
        for x in (x̂ + 1):params[:x_max]
            product = prod(C[k] for k in (x̂ + 1):x)

            upper = (x̂ / x / product)^(1 / (x - x̂))

            if !between(g₁, lower, upper)
                return false
            end
        end
        return true
    end

    """
        condition_mc_bounds(x̂, W, params)

    Diagnostic companion to `condition_mc`: instead of returning a single
    True/False, returns the actual `lower` bound and every `upper` bound
    computed in `condition_mc`'s loop (one per x from x̂+1 to x_max), plus
    their minimum (the binding upper bound). Useful for plotting
    lower/min_upper against g(1) to see *why* g(1) does or doesn't satisfy
    condition_mc, and how close it is.

    Returns a NamedTuple: (lower = ..., x_vals = x̂+1:x_max, uppers = [...], min_upper = ...)
    """
    function condition_mc_bounds(x̂, W, params)
        C = calculate_C(W, params)

        lower = (x̂ - 1) / (x̂ * C[x̂] * C[x̂ - 1])

        x_vals = (x̂ + 1):params[:x_max]
        uppers = Float64[]
        for x in x_vals
            product = prod(C[k] for k in (x̂ + 1):x)
            upper = (x̂ / x / product)^(1 / (x - x̂))
            push!(uppers, upper)
        end
        min_upper = isempty(uppers) ? Inf : minimum(uppers)

        return (lower = lower, x_vals = x_vals, uppers = uppers, min_upper = min_upper)
    end



"""
    check_condition_mc_versus_param(paramkey, paramvec, params_base_orig; kwargs...)

For each value in `paramvec`, solves for the equilibrium group size distribution
the same way `make_hm_versus_param_solve_g` does (same `W_fun`, `scale_fun`,
`equilibria_fun`, `dg_fun`, and ODE refinement step), then checks whether g(1)
at that equilibrium satisfies `condition_mc` (Lemma \\ref{result_condition_mc}
/ \\ref{result_condition_optimal} in the appendix).

x̂, the Clark & Mangel prediction, is computed exactly as in
`make_hm_versus_param_solve_g`: the first x such that W(x+1) < W(1)
(x_max if no such x exists).

Accepts the same keyword arguments as `make_hm_versus_param_solve_g`
(`t_f`, `W_fun`, `scale_fun`, `g0_births`, `dg_fun`, `equilibria_fun`) so that
it can be pointed at the exact same parameter set used to build a given panel.

Returns a NamedTuple:
    g1_vec      : g(1) at equilibrium, for each paramvec value
    xhat_vec    : x̂ for each paramvec value
    satisfied   : Union{Bool,Missing} vector; `missing` when x̂ < 2, since
                  condition_mc's lower bound (which uses C(x̂-1)) isn't
                  defined there.
"""
function check_condition_mc_versus_param(
    paramkey, paramvec, params_base_orig;
    t_f = 50000.0,
    W_fun = (x, p) -> fun_W_orig(x, p[:M1], p[:M2], p),
    scale_fun = scale_parameters_orig,
    g0_births = nothing,
    dg_fun = fun_dg_births_givenW!,
    equilibria_fun = get_g_equilibria_givenW
    )

    @unpack P, x_max = params_base_orig

    xvec = 1:x_max

    params = deepcopy(params_base_orig)
    g1_vec = similar(paramvec, Float64)
    xhat_vec = Vector{Int}(undef, length(paramvec))
    satisfied = Vector{Union{Bool,Missing}}(undef, length(paramvec))

    for (i, param) in enumerate(paramvec)
        params[paramkey] = param
        scale_fun(params)

        W = W_fun(xvec, params)
        params[:Wvec] = W
        gvec = equilibria_fun(P, W, params)

        g0 = (!isnothing(g0_births) && get(params, :allow_births, 0) != 0) ? g0_births : gvec[1:end-1]
        prob = ODEProblem(dg_fun, g0, (0.0, t_f), params)
        sol = solve(prob)
        g_short = sol[:, end]
        g_xmax = (P - sum((1:x_max-1) .* g_short)) / x_max
        gvec = vcat(g_short, g_xmax)

        x̂ = any(W[2:end] .< W[1]) ? findfirst(W[2:end] .< W[1]) : x_max

        g1_vec[i] = gvec[1]
        xhat_vec[i] = x̂
        satisfied[i] = x̂ >= 2 ? condition_mc(gvec[1], x̂, W, params) : missing
    end

    return (g1_vec = g1_vec, xhat_vec = xhat_vec, satisfied = satisfied)
end

"""
    plot_condition_mc_versus_param(paramkey, paramvec, results; kwargs...)

Plots the `satisfied` vector returned by `check_condition_mc_versus_param`
versus `paramvec`, as a 0/1 step plot (missing values are left as gaps), so it
can be lined up underneath the corresponding equilibrium heatmap panel.
"""
function plot_condition_mc_versus_param(paramkey, paramvec, results; kwargs...)
    y = [ismissing(s) ? NaN : Float64(s) for s in results.satisfied]
    plt = plot(paramvec, y;
        seriestype = :steppost,
        ylim = (-0.1, 1.1),
        yticks = ([0, 1], ["fails", "satisfies"]),
        xlabel = param_label_dic[paramkey],
        ylabel = "condition_mc",
        legend = false,
        kwargs...
    )
    return plt
end

"""
    check_condition_mc_bounds_versus_param(paramkey, paramvec, params_base_orig; kwargs...)

Diagnostic version of `check_condition_mc_versus_param`: for each value in
`paramvec`, solves for the equilibrium the same way (same kwargs, same
meaning), then instead of just True/False, records g(1), x̂, the lower bound,
and the *minimum* upper bound (the binding one, over x = x̂+1,...,x_max) from
`condition_mc_bounds`. Plot g1 against `lower` and `min_upper` to see why
g(1) does or doesn't satisfy condition_mc for a given paramvec value.

Returns a NamedTuple:
    g1_vec        : g(1) at equilibrium, for each paramvec value
    xhat_vec      : x̂ for each paramvec value
    lower_vec     : the lower bound, for each paramvec value (NaN when x̂ < 2)
    min_upper_vec : the binding (minimum) upper bound, for each paramvec value (NaN when x̂ < 2)
"""
function check_condition_mc_bounds_versus_param(
    paramkey, paramvec, params_base_orig;
    t_f = 50000.0,
    W_fun = (x, p) -> fun_W_orig(x, p[:M1], p[:M2], p),
    scale_fun = scale_parameters_orig,
    g0_births = nothing,
    dg_fun = fun_dg_births_givenW!,
    equilibria_fun = get_g_equilibria_givenW
    )

    @unpack P, x_max = params_base_orig

    xvec = 1:x_max

    params = deepcopy(params_base_orig)
    g1_vec = similar(paramvec, Float64)
    xhat_vec = Vector{Int}(undef, length(paramvec))
    lower_vec = fill(NaN, length(paramvec))
    min_upper_vec = fill(NaN, length(paramvec))

    for (i, param) in enumerate(paramvec)
        params[paramkey] = param
        scale_fun(params)

        W = W_fun(xvec, params)
        params[:Wvec] = W
        gvec = equilibria_fun(P, W, params)

        g0 = (!isnothing(g0_births) && get(params, :allow_births, 0) != 0) ? g0_births : gvec[1:end-1]
        prob = ODEProblem(dg_fun, g0, (0.0, t_f), params)
        sol = solve(prob)
        g_short = sol[:, end]
        g_xmax = (P - sum((1:x_max-1) .* g_short)) / x_max
        gvec = vcat(g_short, g_xmax)

        x̂ = any(W[2:end] .< W[1]) ? findfirst(W[2:end] .< W[1]) : x_max

        g1_vec[i] = gvec[1]
        xhat_vec[i] = x̂

        if x̂ >= 2
            bounds = condition_mc_bounds(x̂, W, params)
            lower_vec[i] = bounds.lower
            min_upper_vec[i] = bounds.min_upper
        end
    end

    return (g1_vec = g1_vec, xhat_vec = xhat_vec, lower_vec = lower_vec, min_upper_vec = min_upper_vec)
end

"""
    plot_condition_mc_bounds_versus_param(paramkey, paramvec, results; kwargs...)

Plots g(1), the lower bound, and the (binding) minimum upper bound, all
against `paramvec`, on a log y-axis (these quantities can span many orders
of magnitude). g(1) between the two bound curves means condition_mc is
satisfied at that paramvec value.
"""
function plot_condition_mc_bounds_versus_param(paramkey, paramvec, results; kwargs...)
    plt = plot(paramvec, results.g1_vec;
        label = "g(1)",
        yscale = :log10,
        xlabel = param_label_dic[paramkey],
        ylabel = "g(1) / bounds (log scale)",
        legend = :outertopright,
        kwargs...
    )
    plot!(plt, paramvec, results.lower_vec; label = "lower bound")
    plot!(plt, paramvec, results.min_upper_vec; label = "min upper bound")
    return plt
end

end