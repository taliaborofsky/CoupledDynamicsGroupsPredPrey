module ModelFuns
using UnPack
include("ModelHelperFuns.jl")
using .ModelHelperFuns

export fun_dg_nopop!, fun_dg_nopop

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
    g_x_max = P - sum(@. xvec[1:end-1] * g[1:end])
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



end
