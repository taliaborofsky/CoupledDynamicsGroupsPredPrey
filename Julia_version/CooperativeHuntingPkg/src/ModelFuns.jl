module ModelFuns
using UnPack
include("ModelHelperFuns.jl")
using .ModelHelperFuns

export fun_dg_nopop!, fun_dg_nopop
export fun_dg_births_givenW!
export fun_dg_births_givenW_grouplevel!
export fun_dg_births_constantP!, fun_dg_births_constantP
export fun_W_gauss
export fun_dg_simpleW!, fun_dg_simpleW

"""
fun_dg_nopop
finds dg(1)/dT, dg(2)/dT, ..., dg(x_max - 1)/dT
g is g(1), g(2), ..., g(x_max - 1)

params has :N1, :N2, :P in addition to normal parameters
"""
function fun_dg_nopop!(dg, g, params, T)
    # Unpack basic ingredients
    @unpack x_max, N1, N2, P, Tg = params
    xvec = 1:x_max
    # Compute fitnesses and best response functions
    Wvec = fun_W(xvec, N1, N2, params)
    S_1_x = fun_S_given_W(Wvec[1], Wvec, params)
    S_x_1 = @. 1 - S_1_x
    g_x_max = (P - sum(@. xvec[1:end-1] * g[1:end])) ./ x_max
    g = [g..., g_x_max]
    for x in 1:(x_max-1)
        if x == 1
            groups_2_split = 4 * g[2] * S_1_x[2]
            leave_larger_grps = sum(@. xvec[3:end] * g[3:end] * S_1_x[3:end])
            join_groups = -(g[1]) * sum(@. g[1:end-1] * S_x_1[2:end])
            dg[1] = (groups_2_split + leave_larger_grps + join_groups)/Tg

        elseif x == 2
            individual_leaves = -2 * g[2] * S_1_x[2]
            pairs_to_threes = -g[2] * g[1] * S_x_1[3]
            threes_to_pairs = 3 * g[3] * S_1_x[3]
            form_dyads = (g[1])^2 * S_x_1[2] / 2
            dg[2] = (individual_leaves + pairs_to_threes + form_dyads + threes_to_pairs)/Tg

        else
            individual_leaves = -x * g[x] * S_1_x[x]
            grows_to_larger_grp = -g[x] * g[1] * S_x_1[x+1]
            smaller_grp_grows_to_x = g[x-1] * g[1] * S_x_1[x]
            larger_grps_shrink = (x+1) * g[x+1] * S_1_x[x+1]
            dg[x] = (individual_leaves + grows_to_larger_grp + smaller_grp_grows_to_x + larger_grps_shrink)/Tg
        end
    end
end

function fun_dg_nopop(g, params, T)
    dg = copy(g)
    fun_dg_nopop!(dg, g, params, T)
    return dg
end

function fun_dg_nopop(g, params)
    return fun_dg_nopop(g, params, 1.0)
end

"""
    fun_dg_births_givenW!(dg, g, params, T=0)

Like fun_dg_births_constantP! but reads Wvec directly from params[:Wvec] instead
of calling fun_W internally.  No scale_parameters call is made; the caller must
precompute Wvec and store it in params before passing to an ODE solver.

Typical use with fun_W_orig:
    scale_parameters_orig(params)
    params[:Wvec] = fun_W_orig(1:params[:x_max], params[:M1], params[:M2], params)
    prob = ODEProblem(fun_dg_births_givenW!, g0, (0.0, t_f), params)
"""
function fun_dg_births_givenW!(dg, g, params, T=0)
    @unpack x_max, Tg, fuse_param, leave_param, Wvec = params

    xvec = 1:x_max
    # remember x_max is P

    P = x_max
    W1 = Wvec[1]
    S_1_x = fun_S_given_W(W1, Wvec, params)
    g_x_max = (P - sum(@. xvec[1:end-1] * g[1:end])) ./ x_max # get g_{x_max}
    g = [g..., g_x_max]
    δ = sum(xvec .* Wvec .* g) ./ P # death rate, is mu in main text

    allow_births = get(params, :allow_births, 0)

    for x in 1:x_max-1
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

        # elseif x == x_max
        #     individual_leaves = -x * g[x] * S_1_x[x] / Tg
        #     smaller_grp_grows_to_xm = g[x - 1] * g[1] * (1 - S_1_x[x]) / Tg
        #     births = (x - 1) * g[x - 1] * Wvec[x - 1]
        #     deaths = -δ * g[x] * x
        #     dg[x] = (
        #         leave_param * individual_leaves + smaller_grp_grows_to_xm +
        #         allow_births * (births + deaths)
        #     )

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
    fun_dg_births_givenW_grouplevel!(dg, g, params, T=0)

Identical to `fun_dg_births_givenW!`, except that individuals base their
join/leave decisions on a comparison of the group's per-capita fecundity
before and after the transition (i.e., "group fitness"), using
`fun_S_grouplevel`, rather than comparing to the fecundity of a solitary
individual, W(1), via `fun_S_given_W`. See the reply to R1's significant
issue #2 (R1-sig2) in the response to reviewers for the corresponding
best-response functions, S_J(x) and S_l(x).

As with `fun_dg_births_givenW!`, the caller must precompute Wvec and store
it in params before passing to an ODE solver.
"""
function fun_dg_births_givenW_grouplevel!(dg, g, params, T=0)
    @unpack x_max, Tg, fuse_param, leave_param, Wvec = params

    xvec = 1:x_max
    # remember x_max is P

    P = x_max
    S_1_x, _ = fun_S_grouplevel(Wvec, params)
    g_x_max = (P - sum(@. xvec[1:end-1] * g[1:end])) ./ x_max # get g_{x_max}
    g = [g..., g_x_max]
    δ = sum(xvec .* Wvec .* g) ./ P # death rate, is mu in main text

    allow_births = get(params, :allow_births, 0)

    for x in 1:x_max-1
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

# moved in from GroupsOnly.jl (they belong with the other dg/dT right-hand
# sides, not with the group-only equilibrium/condition_mc analysis code)

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

function fun_dg_births_constantP(g, params, T=0)
    dg = similar(g)
    fun_dg_births_constantP!(dg, g, params, T)
    return dg
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
    end
end

function fun_dg_simpleW(g, p, T)
    dg = deepcopy(g)
    fun_dg_simpleW!(dg, g, p, T)
    return dg
end

end
