module GroupsOnly

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults

using UnPack
using LinearAlgebra
using ForwardDiff #this should be able to numerically find a jacobian
using Polynomials
using LaTeXStrings
using Plots
using DifferentialEquations

export find_mangel_clark, get_g_equilibria, classify_equilibrium_g
export update_params, bifurcation_g_input, get_x_maximizes_pc_fitness
export heatmap_bif_g
export get_g_equilibria_givenW
export fun_dg_births_constantP!
export fun_dg_births_constantP
export fun_dg_births_constantP_givenW!
export fun_W_gauss
export fun_dg_simpleW!, fun_dg_simpleW
export make_hm_versus_param

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
    :mass_ratio => L"Mass Ratio, $m_2/m_1$"
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
    if all(real_parts .< 0)
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

function fun_W_gauss(x, p)
    @unpack a, x0, σ = p # height, x that maximizes fecundity, standard deviation
    W = a .* exp.( - (x .- x0).^2 ./(2*σ^2))
end


#= add l, phi to p if not there, if p is dict or named tupe =#
ensure_l_phi(p) = p isa AbstractDict ?
    (get!(p, :l, 1); get!(p, :phi, 1.0); p) :
    (; p..., l = get(p, :l, 1.0), phi = get(p, :phi, 1.0))



function fun_dg_simpleW!(dg, g, p, T)
#=
Group dynamics with leaving and singletons "modulated" by a leave_param and fuse_param
Uses a gaussian W
=#
    # p - parameters - has a, x0, σ, Tg, d, and x_max
    
    # if no l, phi in paramater dictionary (called p), then set equal to 1
    p = ensure_l_phi(p)
    # for k in (:leave_paaram, :fuse_param)
    #     get!(p, k, 1)
    # end
    # unpack basic ingredients
    @unpack x_max, Tg, fuse_param, leave_param = p
    xvec = 1:x_max

    # i'll need fitnesses and best response functions
    Wvec = fun_W_gauss(xvec, p) 
    W1 = Wvec[1]
    S_1_x = fun_S_given_W(Wvec[1],Wvec, p)
    for x in xvec
        if x==1
            if x_max == 1
                groups_2_split = 0
                join_groups = 0
                leave_larger_grps = 0
            else
                groups_2_split = 4*g[2]*S_1_x[2]/Tg
                join_groups = -(g[1]/Tg)*sum(g[2:end-1].*(1 .- S_1_x[3:end]))
                join_groups_singletonsfuse = - (g[1]./ Tg) .*g[1] .* (1 .- S_1_x[2]) 
                if x_max > 2
                    leave_larger_grps = sum(xvec[3:end].*g[3:end].*S_1_x[3:end])/Tg
                else
                    leave_larger_grps = 0
                end
            end
            dg[1] = (leave_param * (groups_2_split + leave_larger_grps) + join_groups 
                        + fuse_param * join_groups_singletonsfuse)

        elseif x == 2
            individual_leaves = - 2*g[2]*S_1_x[2]/Tg
            if x_max == 2
                threes_to_pairs = 0
                pairs_to_threes = 0
            else
                pairs_to_threes = - g[2]*g[1]*(1-S_1_x[3])/Tg
                threes_to_pairs = 3*g[3]*S_1_x[3]/Tg
                # deaths = td * (3*g[3] - 2*g[2])
                # births = g[1]*Wvec[1] - 2*g[2]*Wvec[2]
            end
            form_dyads = (g[1])^2*(1-S_1_x[2])/(2*Tg)
            dg[2] = (leave_param * (individual_leaves + threes_to_pairs) 
                    + pairs_to_threes + fuse_param * form_dyads 
                )

        elseif x == x_max
            individual_leaves = - x*g[x]*S_1_x[x]/Tg
            smaller_grp_grows_to_xm  = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            dg[x] = (leave_param * individual_leaves + smaller_grp_grows_to_xm )

        else
            individual_leaves = -(x/Tg)*g[x]*S_1_x[x]
            grows_to_larger_grp = - g[x]*g[1]*(1 - S_1_x[x+1])/Tg
            smaller_grp_grows_to_x = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            larger_grps_shrink = (x+1)*g[x+1]*S_1_x[x+1]/Tg
            dg[x] = (leave_param * (individual_leaves + larger_grps_shrink) 
                    + grows_to_larger_grp + smaller_grp_grows_to_x
                 )
            
        end
        # do something with x 
    end
end

function fun_dg_simpleW(g, p, T)
    dg = deepcopy(g)
    fun_dg_simpleW!(dg, g, p, T)
    return dg
end


function make_hm_versus_param(
    g0, paramkey, paramvec, params_base, ODE_fun_handle, W_fun_handle;
    t_f = 500)
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

function fun_dg_births_constantP!(dg, g, params, T=0)
    """
    Finds dg/dt if there are births and deaths but the population size stays constant
    N1, N2 (essential for calculating W) given in params
    In place
    """
    params = scale_parameters(params)

    @unpack x_max, Tg, fuse_param, leave_param, N1, N2 = params

    xvec = 1:x_max
    Wvec = fun_W(xvec,N1, N2,params)

    W1 = Wvec[1]
    S_1_x = fun_S_given_W(W1, Wvec, params)
    P = get_p(g, x_max)
    δ = sum(xvec .* Wvec .* g) ./ P

    # find out if allow births and deaths
    allow_births = get(params, :allow_births, 0)
    @unpack allow_births = params

    for x in xvec
        if x == 1
            if x_max == 1
                groups_2_split = 0
                join_groups = 0
                leave_larger_grps = 0
                births = g[1] * Wvec[1]
                deaths = -δ * g[1]
            else
                groups_2_split = 4 * g[2] * S_1_x[2] / Tg
                join_groups = -(g[1] / Tg) * sum(g[2:end-1] .* (1 .- S_1_x[3:end]))
                join_groups_singletonsfuse = -(g[1] / Tg) * g[1] * (1 .- S_1_x[2])
                births = x_max * g[x_max] * Wvec[x_max] - g[1] * Wvec[1]
                deaths = 2 * δ * g[2] - δ * g[1]
                if x_max > 2
                    leave_larger_grps = sum(xvec[3:end] .* g[3:end] .* S_1_x[3:end]) / Tg
                else
                    leave_larger_grps = 0
                end
            end
            dg[1] = (leave_param * (groups_2_split + leave_larger_grps) + join_groups +
                    fuse_param * join_groups_singletonsfuse + 
                    allow_births * (births + deaths))

        elseif x == 2
            individual_leaves = -2 * g[2] * S_1_x[2] / Tg
            if x_max == 2
                threes_to_pairs = 0
                pairs_to_threes = 0
                deaths = -2 * δ * g[2]
                births = g[1] * Wvec[1]
            else
                pairs_to_threes = -g[2] * g[1] * (1 - S_1_x[3]) / Tg
                threes_to_pairs = 3 * g[3] * S_1_x[3] / Tg
                deaths = δ * (3 * g[3] - 2 * g[2])
                births = g[1] * Wvec[1] - 2 * g[2] * Wvec[2]
            end
            form_dyads = g[1]^2 * (1 - S_1_x[2]) / (2 * Tg)
            dg[2] = (leave_param * (individual_leaves + threes_to_pairs) +
                    pairs_to_threes + fuse_param * form_dyads + 
                    allow_births*(births + deaths))

        elseif x == x_max
            individual_leaves = -x * g[x] * S_1_x[x] / Tg
            smaller_grp_grows_to_xm = g[x - 1] * g[1] * (1 - S_1_x[x]) / Tg
            births = (x - 1) * g[x - 1] * Wvec[x - 1]
            deaths = -δ * g[x] * x
            dg[x] = (
                leave_param * individual_leaves + smaller_grp_grows_to_xm + 
                allow_births* (births + deaths)
            )

        else
            individual_leaves = -(x / Tg) * g[x] * S_1_x[x]
            grows_to_larger_grp = -g[x] * g[1] * (1 - S_1_x[x + 1]) / Tg
            smaller_grp_grows_to_x = g[x - 1] * g[1] * (1 - S_1_x[x]) / Tg
            larger_grps_shrink = (x + 1) * g[x + 1] * S_1_x[x + 1] / Tg
            births = (x - 1) * g[x - 1] * Wvec[x - 1] - x * g[x] * Wvec[x]
            deaths = δ * ((x + 1) * g[x + 1] - x * g[x])
            dg[x] = (
                leave_param * (individual_leaves + larger_grps_shrink) +
                    grows_to_larger_grp + smaller_grp_grows_to_x + 
                    allow_births * (births + deaths)
            )
        end
    end
end

"""
    fun_dg_births_constantP_givenW!(dg, g, params, T=0)

Like fun_dg_births_constantP! but reads Wvec directly from params[:Wvec] instead
of calling fun_W internally.  No scale_parameters call is made; the caller must
precompute Wvec and store it in params before passing to an ODE solver.

Typical use with fun_W_orig:
    scale_parameters_orig(params)
    params[:Wvec] = fun_W_orig(1:params[:x_max], params[:M1], params[:M2], params)
    prob = ODEProblem(fun_dg_births_constantP_givenW!, g0, (0.0, t_f), params)
"""
function fun_dg_births_constantP_givenW!(dg, g, params, T=0)
    @unpack x_max, Tg, fuse_param, leave_param, Wvec = params

    xvec = 1:x_max

    W1 = Wvec[1]
    S_1_x = fun_S_given_W(W1, Wvec, params)
    P = get_p(g, x_max)
    δ = sum(xvec .* Wvec .* g) ./ P # death rate, is mu in main text

    allow_births = get(params, :allow_births, 0)

    for x in xvec
        if x == 1
            if x_max == 1
                groups_2_split = 0
                join_groups = 0
                leave_larger_grps = 0
                births = g[1] * Wvec[1]
                deaths = -δ * g[1]
            else
                groups_2_split = 4 * g[2] * S_1_x[2] / Tg
                join_groups = -(g[1] / Tg) * sum(g[2:end-1] .* (1 .- S_1_x[3:end]))
                join_groups_singletonsfuse = -(g[1] / Tg) * g[1] * (1 .- S_1_x[2])
                births = x_max * g[x_max] * Wvec[x_max] - g[1] * Wvec[1]
                deaths = 2 * δ * g[2] - δ * g[1]
                if x_max > 2
                    leave_larger_grps = sum(xvec[3:end] .* g[3:end] .* S_1_x[3:end]) / Tg
                else
                    leave_larger_grps = 0
                end
            end
            dg[1] = (leave_param * (groups_2_split + leave_larger_grps) + join_groups +
                    fuse_param * join_groups_singletonsfuse +
                    allow_births * (births + deaths))

        elseif x == 2
            individual_leaves = -2 * g[2] * S_1_x[2] / Tg
            if x_max == 2
                threes_to_pairs = 0
                pairs_to_threes = 0
                deaths = -2 * δ * g[2]
                births = g[1] * Wvec[1]
            else
                pairs_to_threes = -g[2] * g[1] * (1 - S_1_x[3]) / Tg
                threes_to_pairs = 3 * g[3] * S_1_x[3] / Tg
                deaths = δ * (3 * g[3] - 2 * g[2])
                births = g[1] * Wvec[1] - 2 * g[2] * Wvec[2]
            end
            form_dyads = g[1]^2 * (1 - S_1_x[2]) / (2 * Tg)
            dg[2] = (leave_param * (individual_leaves + threes_to_pairs) +
                    pairs_to_threes + fuse_param * form_dyads +
                    allow_births * (births + deaths))

        elseif x == x_max
            individual_leaves = -x * g[x] * S_1_x[x] / Tg
            smaller_grp_grows_to_xm = g[x - 1] * g[1] * (1 - S_1_x[x]) / Tg
            births = (x - 1) * g[x - 1] * Wvec[x - 1]
            deaths = -δ * g[x] * x
            dg[x] = (
                leave_param * individual_leaves + smaller_grp_grows_to_xm +
                allow_births * (births + deaths)
            )

        else
            individual_leaves = -(x / Tg) * g[x] * S_1_x[x]
            grows_to_larger_grp = -g[x] * g[1] * (1 - S_1_x[x + 1]) / Tg
            smaller_grp_grows_to_x = g[x - 1] * g[1] * (1 - S_1_x[x]) / Tg
            larger_grps_shrink = (x + 1) * g[x + 1] * S_1_x[x + 1] / Tg
            births = (x - 1) * g[x - 1] * Wvec[x - 1] - x * g[x] * Wvec[x]
            deaths = δ * ((x + 1) * g[x + 1] - x * g[x])
            dg[x] = (
                leave_param * (individual_leaves + larger_grps_shrink) +
                    grows_to_larger_grp + smaller_grp_grows_to_x +
                    allow_births * (births + deaths)
            )
        end
    end
end

function fun_dg_births_constantP(g, params, T=0)
    dg = similar(g)
    fun_dg_births_constantP!(dg, g, params, T)
    return dg
end



end