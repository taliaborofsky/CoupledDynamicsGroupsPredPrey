module CooperativeHuntingPkg

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("MyBifTools.jl")
include("ApparentCompTools.jl")
include("GroupsOnly.jl")

using .ModelHelperFuns
using .ModelFuns
using .MyBifTools
using .ApparentCompTools
using .GroupsOnly

# ModelHelperFuns
export scale_parameters, scale_parameters2
export fun_alpha1
export fun_W, fun_S_given_W
export fun_W_orig, scale_parameters_orig
export fun_S_grouplevel
export get_p, get_meanx, get_prob_in_x

# ModelFuns
export fun_dg_nopop!, fun_dg_nopop
export fun_dg_births_givenW!
export fun_dg_births_givenW_grouplevel!
export fun_dg_births_constantP!, fun_dg_births_constantP
export fun_W_gauss
export fun_dg_simpleW!, fun_dg_simpleW

# MyBifTools
export plot_segments!

# ApparentCompTools
export Jacobian_g

# Groups Only (also includes the group-size-distribution analysis
# functions formerly in the now-removed AnalyzeResults.jl)
export find_mangel_clark, get_g_equilibria, classify_equilibrium_g
export update_params, bifurcation_g_input, get_x_maximizes_pc_fitness
export heatmap_bif_g
export get_g_equilibria_givenW
export get_g_equilibria_givenW_grouplevel
export make_hm_versus_param
export make_hm_versus_param_solve_g
# advisor said the condition_mc / Mangel-Clark-recovery check isn't useful --
# the corresponding code is commented out in GroupsOnly.jl, so these exports
# are commented out too.
# export condition_mc, condition_mc_bounds
# export check_condition_mc_versus_param
# export plot_condition_mc_versus_param
# export check_condition_mc_bounds_versus_param
# export plot_condition_mc_bounds_versus_param
# export test_g_ranking_within_condition_mc_bounds
# export summarize_g_ranking_test

end
