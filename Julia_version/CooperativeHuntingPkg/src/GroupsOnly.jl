module GroupsOnly

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
using .ModelHelperFuns
using .ModelFuns

using UnPack
using LinearAlgebra
using ForwardDiff #this should be able to numerically find a jacobian
using Polynomials
using LaTeXStrings
using Plots
using DifferentialEquations
using Measures

# note: fun_dg_births_constantP!, fun_dg_births_constantP, fun_W_gauss,
# fun_dg_simpleW!, fun_dg_simpleW now live in ModelFuns.jl (imported via
# `using .ModelFuns` above); get_p, get_meanx, get_prob_in_x now live in
# ModelHelperFuns.jl (imported via `using .ModelHelperFuns` above).
# AnalyzeResults.jl has otherwise been merged into this module
# (condition_mc, condition_mc_bounds, calculate_C, between are defined below).

export find_mangel_clark, get_g_equilibria, classify_equilibrium_g
export update_params, bifurcation_g_input, get_x_maximizes_pc_fitness
export heatmap_bif_g
export get_g_equilibria_givenW
export get_g_equilibria_givenW_grouplevel
export make_hm_versus_param
export make_hm_versus_param_solve_g
# advisor said the condition_mc / Mangel-Clark-recovery check isn't useful --
# commented out below (calculate_C, between, condition_mc, condition_mc_bounds,
# check_/plot_condition_mc_*_versus_param, test_g_ranking_within_condition_mc_bounds,
# summarize_g_ranking_test), so these exports are commented out too.
# export condition_mc, condition_mc_bounds
# export check_condition_mc_versus_param
# export plot_condition_mc_versus_param
# export check_condition_mc_bounds_versus_param
# export plot_condition_mc_bounds_versus_param
# export test_g_ranking_within_condition_mc_bounds
# export summarize_g_ranking_test


# these were supposed to load with my package but i guess it didn't work
ylabel_dic = Dict(
    :N1 => L"N_1"*", Scaled Big Prey\nDensity",
    :N2 => L"N_2"*", Scaled Small Prey\nDensity",
    :mean_x => "Mean Experienced\nGroup Size, "*L"\bar{x}",
    :p => L"Predator density, $p$",
    :Nsum => "Sum of Prey Densities,\n"*L"N_1 + N_2",
    :g1 => L"g(1)",
    :g2 => L"g(2)",
    :g3 => L"g(3)"
)

param_label_dic = Dict(
    :α1_of_1 => L"\alpha_1(1)",
    :α2_of_1 => L"\alpha_2(1)",
    :s1 => L"s_1",
    :s2 => L"s_2",
    :H1a => L"H_{1a}",
    :H2a => L"H_{2a}",
    :H2b => L"H_{2b}",
    :A1 => L"Relative Attack Rate of Big Prey, $A_1$",
    :A2 => L"A_2",
    :η2 => L"Growth of Small Prey, $\eta_2$",
    :β2 => L"\beta_2",
    :x_max => L"x_{max}",
    :Tg => "Relative Group Dynamics \nTimescale, "*L"T_g",
    :d => L"d",
    :scale => L"Scale, $\beta_1/\beta_2$",
    :mass_ratio => L"Mass Ratio, $m_2/m_1$",
    :σ => L"Standard Deviation of $W$, $\sigma$",
    :a => L"Maximum Fitness Height, $a$",
    :x0 => L"Group Size that Maximizes Fitness, $x^*$"
)

function find_mangel_clark(N1, N2, params)
    # Mangel and Clark predicted that groups should grow until W(x^*) = W(1)
    # Simplest way: iterate and stop when W(x) < W(1), then return x - 1
    @unpack x_max = params
    W_of_1 = fun_W(1, N1, N2, params)
    for x in 2:x_max
        W_of_x = fun_W(x, N1, N2, params)
        if W_of_x < W_of_1
            return x - 1
        end
    end
    return x_max  # If reach x_max
end



function get_g_equilibria(P, N1, N2, params)
    """
    Finds all the g(x) equilibria for a certain p, N1, N2 combination.
    Assumes population sizes are constant.

    Returns gvec

    THIS MIGHT NOT BE USED ANYMORE
    """
    x_max = params[:x_max]
    xvec = 1:x_max

    # Get the root for g(1)

    W = fun_W(xvec,N1,N2,params)
    S_of_1_x = fun_S_given_W(W[1], W, params)
    S_of_x_1 = 1 .- S_of_1_x
    c_vec = S_of_x_1./(xvec.*S_of_1_x)
    c_vec[2] = c_vec[2]/2    
    # Compute coefficients for g(1)
    coefficients = [x * prod(c_vec[1:x]) for x in xvec]  # Reverse order
    coeff_full = vcat(-P, coefficients)  # Append -P to the coefficients
    # Find roots
    roots_all = roots(Polynomials.Polynomial(coeff_full))
    
    # Filter real positive roots. there should only be one.
    g1 = real(filter(x -> isreal(x) && real(x) > 0, roots_all)[1])

    # Compute g(x) for each g1 root
    gvec = [prod(c_vec[1:x]) * g1^x for x in xvec]

    return gvec
end


function classify_equilibrium_g(g, N1, N2, params)
    """
    Compute the eigenvalues of the Jacobian matrix for just the dynamics of dg(x)/dt.
    Returns:
    - "Stable (attractive)"
    - "Unstable"
    - "Marginally stable (needs further analysis)"
    - "Indeterminate stability (needs further analysis)"
    """
    # Compute the Jacobian matrix for group dynamics
    J = ForwardDiff.jacobian(u -> fun_dg_nopop(u, params, 1),g[1:end-1])
    #J = Jacobian_g(N1, N2, g, params)

    # Compute the eigenvalues of the Jacobian matrix
    eigenvalues = eigen(J).values
    # Check the real parts of the eigenvalues
    real_parts = real.(eigenvalues)

    # Classify the stability based on the real parts of the eigenvalues
    if all(real_parts .<= 1e-9) # classifying marginally stable as stable
        return true
    else
        return false
    end
end
function update_params(paramkey::Symbol, param, params_base::Dict{Symbol, Any})
    # Create a copy of the base parameters
    params = deepcopy(params_base)
    
    # Update the parameter specified by paramkey with the new value
    params[paramkey] = param
    
    # Scale the parameters
    params = scale_parameters(params)
    
    return params
end

function bifurcation_g_input(p, N1, N2, paramkey::Symbol, 
    paramvec, params_base::Dict{Symbol, Any})
    #=
    Loop over elements of paramvec, 
    finding the g equilibrium and stability for each
    paramater value
    =#
    x_max = params_base[:x_max]
    # Initialize arrays to store equilibrium g values and their corresponding stability
    results_g = zeros(length(paramvec), x_max)  # 2D array
    stability_results = Vector{Bool}(undef, length(paramvec))
    
    
    # Iterate over paramvec
    for (i, param) in enumerate(paramvec)
        # Update parameters
        params = update_params(paramkey, param, params_base)

        # Find the single equilibrium g vector
        gvec = get_g_equilibria(p, N1, N2, params)

        # Get stability
        stability = classify_equilibrium_g(gvec, N1, N2, params)

        # Store results
        results_g[i, :] = gvec  # Store the vector as a row
        stability_results[i] = stability
    end

    return Dict(
        :results_g => results_g, 
        :stability_results => stability_results
        )
end

function get_x_maximizes_pc_fitness(N1, N2, params)
    xvec = 1:params[:x_max]
    W_of_x = fun_W(xvec, N1, N2, params)  # Use fun_W instead of per_capita_fitness_from_prey_non_dim
    max_index = argmax(W_of_x)
    return max_index
end


# stuff for a simple Gaussian W

function get_g_equilibria_givenW(P, W, params)
    """
    Finds all the g(x) equilibria for a certain p, N1, N2 combination.
    Assumes population sizes are constant.

    Returns gvec
    """
    x_max = params[:x_max]
    xvec = 1:x_max

    # if no l, phi in params, then set equal to 1
    for k in (:leave_param, :fuse_param)
        get!(params, k, 1)
    end

    @unpack leave_param,fuse_param = params
    l, ϕ = leave_param, fuse_param
    # Get the root for g(1)

    S_of_1_x = fun_S_given_W(W[1], W, params)
    S_of_x_1 = 1 .- S_of_1_x
    c_vec = S_of_x_1./(l .* xvec.* S_of_1_x)
    c_vec[1] = 1.0

    # Compute coefficients for g(1)
    coefficients = [x * ϕ/2 * prod(c_vec[1:x]) for x in xvec]  # Reverse order
    coefficients[1] = 1.0 # the coefficient of g_1 

    coeff_full = vcat(-P, coefficients)  # Append -P to the coefficients
    # Find roots
    roots_all = roots(Polynomials.Polynomial(coeff_full))

    # Filter real positive roots. there should only be one.
    g1 = real(filter(x -> isreal(x) && real(x) > 0, roots_all)[1])

    # Compute g(x) for each g1 root
    gvec = [0.5 * ϕ * prod(c_vec[1:x]) * g1^x for x in xvec]
    gvec[1] = g1

    return gvec
end

"""
    get_g_equilibria_givenW_grouplevel(P, W, params)

Same as `get_g_equilibria_givenW`, but for the version of the model in which
individuals compare the group's fecundity before and after joining/leaving
(i.e., "group fitness") rather than comparing to the fecundity of a
solitary individual. Uses `fun_S_grouplevel` instead of `fun_S_given_W`.

The closed-form relationship g_x = g_1 g_{x-1} * S_J(x) / (l * x * (1-S_J(x)))
(SI eq. `gx_interms_gxminus1`) holds regardless of how S_J(x) is defined, so
the rest of the derivation (and this function) is unchanged.

Returns gvec.
"""
function get_g_equilibria_givenW_grouplevel(P, W, params)
    x_max = params[:x_max]
    xvec = 1:x_max

    # if no l, phi in params, then set equal to 1
    for k in (:leave_param, :fuse_param)
        get!(params, k, 1)
    end

    @unpack leave_param,fuse_param = params
    l, ϕ = leave_param, fuse_param
    # Get the root for g(1)

    S_of_1_x, S_of_x_1 = fun_S_grouplevel(W, params)
    c_vec = S_of_x_1./(l .* xvec.* S_of_1_x)
    c_vec[1] = 1.0

    # Compute coefficients for g(1)
    coefficients = [x * ϕ/2 * prod(c_vec[1:x]) for x in xvec]  # Reverse order
    coefficients[1] = 1.0 # the coefficient of g_1

    coeff_full = vcat(-P, coefficients)  # Append -P to the coefficients
    # Find roots
    roots_all = roots(Polynomials.Polynomial(coeff_full))

    # Filter real positive roots. there should only be one.
    g1 = real(filter(x -> isreal(x) && real(x) > 0, roots_all)[1])

    # Compute g(x) for each g1 root
    gvec = [0.5 * ϕ * prod(c_vec[1:x]) * g1^x for x in xvec]
    gvec[1] = g1

    return gvec
end

function make_hm_versus_param(
    g0, paramkey, paramvec, params_base, ODE_fun_handle, W_fun_handle;
    t_f = 50000)
    #=
    Plot a heatmap of Pr(x) versus some parameter that's being varied.
    ODE_fun_handle gives the function used to calculate dg and must be in-place.
    Keeps population size constant.

    This works without any scaling of parameters that has to be done every time
    the params dictionary is updated.
    =#
    @unpack x_max = params_base
    dg = zeros(x_max)
    P = get_p(g0, x_max)
    params = deepcopy(params_base)

    n = length(paramvec)
    pxmat = zeros(n, x_max)
    reached_equilibrium_vec = zeros(n)
    mean_x_vec = zeros(n)
    mc_x_vec = zeros(n)
    x_opt_vec = similar(paramvec)

    for (i, param) in enumerate(paramvec)
        params[paramkey] = param

        if ODE_fun_handle == fun_dg_simpleW!
            W = fun_W_gauss(1:x_max, params)
            g_final = get_g_equilibria_givenW(P, W, params)
        elseif ODE_fun_handle == fun_dg_births_constantP! 
                params = scale_parameters(params)
                prob = ODEProblem(ODE_fun_handle, g0, (0, t_f), NamedTuple(params))
                sol = solve(prob)
                g_final = sol[:,end]
        else
            prob = ODEProblem(ODE_fun_handle, g0, (0, t_f), NamedTuple(params))
            sol = solve(prob)
            g = sol.u
            g_final = g[end]
        end

        px = get_prob_in_x(g_final, P, x_max)
        pxmat[i, :] = px

        ODE_fun_handle(dg, g_final, NamedTuple(params), 1.0)
        reached_equilibrium_vec[i] = all(abs.(dg) .< 1e-10)
        mean_x_vec[i] = get_meanx(g_final, x_max)

        W = W_fun_handle(1:x_max, params)
        mc_x_vec[i] = findlast(>=(W[1]), W)
        x_opt_vec[i] = argmax(W)

        g0 = g_final
    end

    xlabel_dict = Dict(
        :σ => L"Standard Deviation of $W$, $\sigma$",
        :a => L"Maximum Fitness Height, $a$",
        :x0 => L"Group Size that Maximizes Fitness, $x_0$",
        :scale => L"Benefit Ratio, $\beta_1/\beta_2$"
    )

    hm = heatmap(
        paramvec,
        1:size(pxmat, 2),
        pxmat',
        c = cgrad([:white, :black], 256),
        clims = (0.0, findmax(pxmat)[1]),
        xlabel = xlabel_dict[paramkey],
        ylabel = L"Group size, $x$",
    )
    plot!(paramvec, mean_x_vec, label = "Mean Experienced", color = :yellow)
    plot!(paramvec, mc_x_vec, label = "Clark and Mangel", color = :limegreen)
    plot!(paramvec, x_opt_vec, label = "Optimal", color = :magenta)

    return hm, reached_equilibrium_vec
end

function heatmap_bif_g(gmat, P::Number, N1::Number, N2::Number, paramkey, paramvec, params_base)
    #=
    uses a heatmap to plot Prob(x), if P, N1, N2 constant
    uses the function fun_W to get W and thus find the highest x for which W(x) >= W(1) (called x^*)
        and the x that maximizes W (called x_0)
    =#

    prob = get_prob_in_x(gmat, P, params_base[:x_max])
    hm = heatmap(
        paramvec,     # x = rows
        1:size(prob,2),     # y = columns
        prob',              # transpose so rows map to x
        c = cgrad([:white, :black], 256),
        #colorrev=false,      # darker = higher
        xlabel=param_label_dic[paramkey],
        ylabel=L"Group size, $x$",
        title="Probability heatmap",
        ylims = [1, params_base[:x_max]]
    )
    # find x_mc
    x_mc_vec = similar(paramvec)
    x_opt_vec = similar(paramvec)
    for (i,param) in enumerate(paramvec)
            params = update_params(paramkey, param, params_base)
            W = fun_W(1:params[:x_max],N1,N2,params)

            # give the index of the first group size x, where x > 1, such that W(x) >=W(1) and W(x+1)<W(1).
            # otherwise (if the fitness is > W(1) for all group sizes) give the maximum group size
            x_mc_vec[i] = any(W[2:end] .< W[1]) ? findfirst( W[2:end] .< W[1]) : params[:x_max]
            x_opt_vec[i] = argmax(W)
    end
    plot!(paramvec, x_mc_vec, label = "Clark & Mangel", color = :limegreen)
    plot!(paramvec, x_opt_vec, label = "Optimal", color = :magenta)
    return hm
end

function plot_hm_with_lines(prob, x_max, paramvec, paramkey,
            x_opt_vec, x_mc_vec, mean_x_vec; offsets = [+0.4, 0.4, -0.4]
        )
    hm = heatmap(
            paramvec,
            1:size(prob, 2),
            prob',
            c = cgrad([:white, :black], 256),
            xlabel = "\n" * param_label_dic[paramkey],
            ylabel = L"Group size, $x$",
            ylims = [1, x_max],
            bottom_margin=5mm
        )
    plot!(paramvec, x_mc_vec, label = "Clark & Mangel", color = :limegreen)
    plot!(paramvec, mean_x_vec, label = "Mean Exp.", color = :yellow)
    plot!(paramvec, x_opt_vec, label = "Optimal", color = :magenta)
    annotate!(paramvec[end-5], x_mc_vec[end] + offsets[1], text("Clark & Mangel", :black, :right, 12))
    annotate!(paramvec[end-5], mean_x_vec[end] + offsets[2], text("Mean Experienced", :black, :right, 12))
    annotate!(paramvec[end-5], x_opt_vec[end] + offsets[3], text("Optimal", :black, :right, 12))
    return hm
end

"""
Makes a heatmap of the probability of being in each group size at equilibria,
in relation to the param specified by paramkey

offsets = [mangel clark offset, mean experienced offset, optimal W offset]

Returns (hm, λ_max_vec) where λ_max_vec[i] is the leading real part of the
Jacobian eigenvalues at the equilibrium for paramvec[i].
"""
function make_hm_versus_param_solve_g(
    paramkey, paramvec, params_base_orig;
    t_f = 50000.0, offsets = [+0.4, 0.4, -0.4],
    W_fun = (x, p) -> fun_W_orig(x, p[:M1], p[:M2], p),
    scale_fun = scale_parameters_orig,
    g0_births = nothing,   # optional starting point for the ODE; defaults to analytical equilibrium
    dg_fun = fun_dg_births_givenW!,   # ODE right-hand side; pass fun_dg_births_givenW_grouplevel!
                                       # for the version where individuals compare group fitness
                                       # before/after joining or leaving, rather than their own
                                       # fitness relative to being solitary
    equilibria_fun = get_g_equilibria_givenW   # analytical equilibrium solver matching dg_fun;
                                                 # pass get_g_equilibria_givenW_grouplevel to match
                                                 # fun_dg_births_givenW_grouplevel!
    )

    @unpack P, x_max = params_base_orig

    xvec = 1:x_max

    results_g = zeros(length(paramvec), x_max)
    params = deepcopy(params_base_orig)
    x_mc_vec = similar(paramvec)
    x_opt_vec = similar(paramvec)
    mean_x_vec = similar(paramvec)
    λ_max_vec = similar(paramvec)

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

        J = ForwardDiff.jacobian(
            g -> (dg = similar(g); dg_fun(dg, g, params, 0); dg),
            gvec[1:end-1]
        )
        λ_max_vec[i] = maximum(real.(eigen(J).values))

        results_g[i, :] = gvec
        x_opt_vec[i] = argmax(W)
        x_mc_vec[i] = any(W[2:end] .< W[1]) ? findfirst(W[2:end] .< W[1]) : x_max
        mean_x_vec[i] = get_meanx(gvec, x_max)
    end

    prob = get_prob_in_x(results_g, P, x_max)

    hm = plot_hm_with_lines(
        prob, x_max, paramvec, paramkey, x_opt_vec, x_mc_vec, mean_x_vec;
        offsets = offsets
    )

    return hm, λ_max_vec
end

# ==============================================================================
# Everything below, until the matching block-comment close near the end of the
# module, checks whether the equilibrium g(1) recovers the Mangel & Clark
# prediction (condition_mc / Lemma result_condition_mc). Commented out:
# advisor's feedback was that this check isn't useful. Left in place (rather
# than deleted) in case it's wanted again.
# ==============================================================================
#=
# Functions for checking whether g1 in bounds so mangel clark condition recovered
    function calculate_C(W, params)
        """
        calculates C(x) given W(x) for each x
        CHECK
        """

        # get l, the leaving paramater
        get!(params, :leave_param, 1)
        l = params[:leave_param]

        # xvec
        xvec = collect(1:params[:x_max])

        # get best response functions
        S_J = fun_S_given_W(W, W[1], params)
        S_L = 1 .- S_J

        # now find C
        C = S_J./(l .* xvec.* S_L)
        C[1] = 1.0
        return C
    end

    function between(x, lower, upper)
        if x isa Number
            return lower < x < upper
        else
            return all((lower .< x) .& (x .< upper))
        end
    end

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

"""
    test_g_ranking_within_condition_mc_bounds(paramkey, paramvec, params_base_orig; kwargs...)

Independent sanity check of `condition_mc` / Lemma \\ref{result_condition_mc}:
for each value in `paramvec` and each `(leave_param, fuse_param)` pair, this
computes x̂ and the `condition_mc_bounds` (lower bound, per-x upper bounds),
samples several g1 values strictly inside `(lower, min_upper)` — i.e. values
for which `condition_mc` would return `true` — builds the *full* equilibrium
group-size vector from each sampled g1 via the closed form
eq. \\ref{equilibrium_groups_g1} (g_x = 0.5*fuse_param*C(x)*C(x-1)*...*C(1)*g1^x,
with g_1 = g1), and checks that x̂*g_x̂ > x*g_x for every other x — i.e. that
x̂ really is the most frequently experienced group size, as the Lemma claims.

Unlike `check_condition_mc_versus_param`, this does NOT solve any ODE or
root-find g1 from P: it directly tests whether condition_mc's bounds are
*sufficient* for the ranking claim, using synthetic g1 values chosen to just
satisfy those bounds (not necessarily an actual equilibrium of the dynamics).

kwargs:
    W_fun            : same meaning as in `make_hm_versus_param_solve_g` (default `fun_W_orig`)
    scale_fun        : same meaning as in `make_hm_versus_param_solve_g` (default `scale_parameters_orig`)
    leave_fuse_pairs : Vector of `(leave_param, fuse_param)` pairs to test;
                       default `[(1.0, 1.0), (0.01, 0.01)]`
    n_g1_samples     : how many g1 values to sample (log-spaced, strictly
                       inside `(lower, min_upper)`) per (pair, paramvec value)

Returns a `Vector` of `NamedTuple`s, one per (leave/fuse pair, paramvec value,
g1 sample) test: `(leave_param, fuse_param, param, xhat, g1, passed, failing_x)`.
`failing_x` is `nothing` when `passed`, otherwise the first x where the
ranking failed. Cases where x̂ < 2, or where `(lower, min_upper)` is empty
(no valid g1), are skipped and not included in the results.
"""
function test_g_ranking_within_condition_mc_bounds(
    paramkey, paramvec, params_base_orig;
    W_fun = (x, p) -> fun_W_orig(x, p[:M1], p[:M2], p),
    scale_fun = scale_parameters_orig,
    leave_fuse_pairs = [(1.0, 1.0), (0.01, 0.01)],
    n_g1_samples = 5
    )

    @unpack x_max = params_base_orig
    xvec = 1:x_max

    results = NamedTuple[]

    for (l, ϕ) in leave_fuse_pairs
        params = deepcopy(params_base_orig)
        params[:leave_param] = l
        params[:fuse_param] = ϕ

        for param in paramvec
            params[paramkey] = param
            scale_fun(params)

            W = W_fun(xvec, params)
            x̂ = any(W[2:end] .< W[1]) ? findfirst(W[2:end] .< W[1]) : x_max
            x̂ < 2 && continue

            bounds = condition_mc_bounds(x̂, W, params)
            bounds.lower >= bounds.min_upper && continue  # no valid g1 in range

            C = calculate_C(W, params)

            # log-spaced samples strictly inside (lower, min_upper)
            log_lo = log(bounds.lower)
            log_hi = log(bounds.min_upper)
            fracs = range(0.1, 0.9; length = n_g1_samples)
            for frac in fracs
                g1 = exp(log_lo + frac * (log_hi - log_lo))

                gvec = [0.5 * ϕ * prod(C[1:x]) * g1^x for x in xvec]
                gvec[1] = g1

                xg = xvec .* gvec
                xg_xhat = xg[x̂]

                failing_x = nothing
                for x in xvec[2:end]
                    x == x̂ && continue
                    if !(xg_xhat > xg[x])
                        failing_x = x
                        break
                    end
                end

                push!(results, (
                    leave_param = l, fuse_param = ϕ, param = param,
                    xhat = x̂, g1 = g1, passed = isnothing(failing_x), failing_x = failing_x
                ))
            end
        end
    end

    return results
end

"""
    summarize_g_ranking_test(results)

Prints a pass/fail summary of `test_g_ranking_within_condition_mc_bounds`'s
output, grouped by `(leave_param, fuse_param)`, and lists (up to 5) failures
per group with enough detail to look them up.
"""
function summarize_g_ranking_test(results)
    pairs = unique((r.leave_param, r.fuse_param) for r in results)
    for (l, ϕ) in pairs
        subset = filter(r -> r.leave_param == l && r.fuse_param == ϕ, results)
        n_total = length(subset)
        n_passed = count(r -> r.passed, subset)
        println("ℓ=$l, φ=$ϕ: $n_passed/$n_total passed")
        failures = filter(r -> !r.passed, subset)
        for r in first(failures, 5)
            println("  FAIL: param=$(r.param), x̂=$(r.xhat), g1=$(r.g1), failing_x=$(r.failing_x)")
        end
        if length(failures) > 5
            println("  ... and $(length(failures) - 5) more failures")
        end
    end
end
=#

end