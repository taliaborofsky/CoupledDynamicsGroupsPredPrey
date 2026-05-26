module CooperativeHuntingPkg

include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
include("MyBifTools.jl")
include("ApparentCompTools.jl")
include("GroupsOnly.jl")

using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults
using .MyBifTools
using .ApparentCompTools
using .GroupsOnly

# ModelHelperFuns
export scale_parameters, scale_parameters2
export fun_alpha1
export fun_W, fun_S_given_W
export fun_W_orig, scale_parameters_orig

# ModelFuns
export fun_dg_nopop!, fun_dg_nopop
export fun_dg_births_givenW!

# AnalyzeResults
export get_p, get_meanx, get_prob_in_x

# MyBifTools
export plot_segments!

# ApparentCompTools
export Jacobian_g

# Groups Only
export find_mangel_clark, get_g_equilibria, classify_equilibrium_g
export update_params, bifurcation_g_input, get_x_maximizes_pc_fitness
export heatmap_bif_g
export get_g_equilibria_givenW
export fun_dg_births_constantP!
export fun_dg_births_constantP
export fun_W_gauss
export fun_dg_simpleW!, fun_dg_simpleW
export make_hm_versus_param
export make_hm_versus_param_solve_g



end
