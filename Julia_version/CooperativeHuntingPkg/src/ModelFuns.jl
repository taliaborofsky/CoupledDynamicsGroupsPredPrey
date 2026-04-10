#using UnPack

module ModelFuns
using UnPack
include("ModelHelperFuns.jl")
using .ModelHelperFuns

export fullsystem_scaled!, fullsystem!, fullsystem_scaled2!, fun_dg!, fun_dN1dT!, fun_dN2dT!
export fun_dg_nopop!, fun_dg_nopop
export system_scaled_nogroups, system_scaled_nogroups!, system_nogroups!
export system_nogroups
export CoupledDynamics_1Prey_ODE, CoupledDynamics_1Prey_ODE!
# insert fullsystem scaled

function CoupledDynamics_1Prey_ODE(u,p,T=0)

    du = @. copy(u)*0


    CoupledDynamics_1Prey_ODE!(du,u,p,T)
    
    return du
end

function CoupledDynamics_1Prey_ODE!(du,u,p,T=0)

    # call groups...  need to adjust du, u

    du_N2 = [du[1], 0, du[2:end]...]
    u_N2 = [u[1],0,u[2:end]...]
    fullsystem_scaled!(du_N2, u_N2, p, T)

    # write results back into du instead of reassigning
    du[1] = du_N2[1]
    du[2:end] .= du_N2[3:end]
    return nothing
end
function fullsystem_scaled2!(du,u,p,T=0)
    newp = scale_parameters2(p) # update parameters
    fullsystem!(du,u,newp,T)
end

function fullsystem_scaled!(du,u,p,T=0)
    newp = scale_parameters(p) # update parameters
    fullsystem!(du,u,newp,T)
end

function fullsystem!(du, u, p, T=0)
    #=
    dN1/dT, dN2/dT, dg1/dT, dg2/dT, ... , dg(xmax)/dT
    u = N1,N2,g(1), g(2), ..., g(xmax)
    p is a named tuple or dictionary of parameters
    T is time (rescaled)
    =#
    @assert length(du) == length(u)
    fun_dN1dT!(du, u, p,T)
    fun_dN2dT!(du, u, p,T)
    fun_dg!(du,u,p,T) # update dg vector
end

function fun_dg!(du, u, p, T)

    # unpack basic ingredients
    @unpack x_max, Tg, η1, η2 = p
    N1, N2 = u[1], u[2]
    g = u[3:end]
    xvec = 1:x_max
    td = 1 - η1 - η2 # tilde{\delta}

    # i'll need fitnesses and best response functions
    Wvec = fun_W(xvec, N1, N2, p) 
    W1 = Wvec[1]
    S_1_x = fun_S_given_W(Wvec[1],Wvec, p)
    dg = du[3:end]
    for x in xvec
        if x==1
            if x_max == 1
                groups_2_split = 0
                join_groups = 0
                leave_larger_grps = 0
                births = g[1]*Wvec[1]
                deaths = - td*g[1]
            else
                groups_2_split = 4*g[2]*S_1_x[2]/Tg
                join_groups = -(g[1]/Tg)*sum(g[1:end-1].*(1 .- S_1_x[2:end]))
                births = x_max*g[x_max]*Wvec[x_max] - g[1]*Wvec[1]
                deaths = 2*td*g[2] - td*g[1]
                if x_max > 2
                    leave_larger_grps = sum(xvec[3:end].*g[3:end].*S_1_x[3:end])/Tg
                else
                    leave_larger_grps = 0
                end
            end
            dg[1] = groups_2_split + leave_larger_grps + join_groups + births + deaths

        elseif x == 2
            individual_leaves = - 2*g[2]*S_1_x[2]/Tg
            if x_max == 2
                threes_to_pairs = 0
                pairs_to_threes = 0
                deaths = -2*td*g[2]
                births = g[1]*Wvec[1]
            else
                pairs_to_threes = - g[2]*g[1]*(1-S_1_x[3])/Tg
                threes_to_pairs = 3*g[3]*S_1_x[3]/Tg
                deaths = td * (3*g[3] - 2*g[2])
                births = g[1]*Wvec[1] - 2*g[2]*Wvec[2]
            end
            form_dyads = (g[1])^2*(1-S_1_x[2])/(2*Tg)
            dg[2] = (individual_leaves + pairs_to_threes + form_dyads + threes_to_pairs
                + births + deaths)

        elseif x == x_max
            individual_leaves = - x*g[x]*S_1_x[x]/Tg
            smaller_grp_grows_to_xm  = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            births = (x-1)*g[x-1]*Wvec[x-1]
            deaths = - td * g[x] * x
            dg[x] = (individual_leaves + smaller_grp_grows_to_xm + births + deaths)

        else
            individual_leaves = -(x/Tg)*g[x]*S_1_x[x]
            grows_to_larger_grp = - g[x]*g[1]*(1 - S_1_x[x+1])/Tg
            smaller_grp_grows_to_x = g[x-1]*g[1]*(1-S_1_x[x])/Tg
            larger_grps_shrink = (x+1)*g[x+1]*S_1_x[x+1]/Tg
            births = (x-1)*g[x-1]*Wvec[x-1] - x*g[x]*Wvec[x]
            deaths = td * ((x+1)*g[x+1] - x*g[x])
            dg[x] = (individual_leaves + grows_to_larger_grp + smaller_grp_grows_to_x
                + larger_grps_shrink + births + deaths)
        end
        # do something with x 
    end
    du[3:end] = dg
end

function system_scaled_nogroups(u,p,T=0)
    du = @. copy(u)*0
    system_scaled_nogroups!(du,u,p,T)
    return du
end

function system_scaled_nogroups!(du, u, p, T)
    newparams = scale_parameters(p)
    system_nogroups!(du,u,newparams,T)
end

function system_nogroups(u,params,T)
    #=
    dPdT, assuming group size is constant (one of the parameters in p)
    =#
    N1,N2,P = u
    @unpack x, η1, η2 = params
    δ = 1 - η1 - η2

    W = fun_W(x, N1, N2, params)
    f1 = fun_f1(x, N1, N2, params)  
    f2 = fun_f2(x, N1, N2, params)  

    du = [η1*N1*(1 - N1) - f1*P/x, η2*N2*(1-N2) - f2*P/x, P * (W - δ)]
    return du
end

function system_nogroups!(du, u,params,T)
    #=
    dPdT, assuming group size is constant (one of the parameters in p)
    =#
    N1,N2,P = u
    @unpack x, η1, η2 = params
    δ = 1 - η1 - η2

    W = fun_W(x, N1, N2, params)
    f1 = fun_f1(x, N1, N2, params)  
    f2 = fun_f2(x, N1, N2, params)  
    du[1] = η1*N1*(1 - N1) - f1*P/x
    du[2] = η2*N2*(1-N2) - f2*P/x
    du[3] = P * (W - δ)
end
function fun_dN1dT!(du, u, p,T)
    #=
    dN1dT, the change in big prey population size versus time, non-dimensionalized.

    Inputs:
    - du: Vector to store the derivatives.
    - u: Vector containing [N1, N2, g_of_x_vec...].
    - p: Dictionary of parameters, must include `:x_max`, `:η1`, and others.

    Updates:
    - du[1]: The derivative of N1 with respect to time.
    =#
    # Extract variables from u
    N1, N2 = u[1], u[2]
    g_of_x_vec = u[3:end]  # g(1), g(2), ..., g(x_max)

    # Extract parameters from p
    @unpack x_max, η1 = p
    @assert length(u) == x_max+2 length(u)


    
    x_vec = 1:x_max
    tildef1_of_x = fun_f1(x_vec, N1, N2, p)  
    du[1] = η1 * N1 * (1 - N1) - sum(@. g_of_x_vec * tildef1_of_x)
    
end

function fun_dN2dT!(du, u, p,T)
    #=
    dN2dT, the change in small prey population size versus time, non-dimensionalized.

    Inputs:
    - du: Vector to store the derivatives.
    - u: Vector containing [N1, N2, g_of_x_vec...].
    - p: Dictionary of parameters, must include `:x_max`, `:η1`, and others.

    =#
    # Extract variables from u
    N1, N2 = u[1], u[2]
    g_of_x_vec = u[3:end]  # g(1), g(2), ..., g(x_max)

    # Extract parameters from p
    @unpack x_max, η2 = p

    
    x_vec = 1:x_max
    tildef2_of_x = fun_f2(x_vec, N1, N2, p)  
    du[2] = η2 * N2 * (1 - N2) - sum(@. g_of_x_vec * tildef2_of_x)
    
end
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

        # elseif x == x_max
        #     individual_leaves = -x * g[x] * S_1_x[x] 
        #     smaller_grp_grows_to_xm = g[x-1] * g[1] * S_x_1[x] 
        #     dg[x] = (individual_leaves + smaller_grp_grows_to_xm)/Tg
        else
            individual_leaves = -x * g[x] * S_1_x[x]
            grows_to_larger_grp = -g[x] * g[1] * S_x_1[x+1]
            smaller_grp_grows_to_x = g[x-1] * g[1] * S_x_1[x]
            larger_grps_shrink = (x+1) * g[x+1] * S_1_x[x+1] 
            dg[x] = (individual_leaves + grows_to_larger_grp + smaller_grp_grows_to_x + larger_grps_shrink)/Tg
        end
    end

    #return dg
end

function fun_dg_nopop(g, params, T)
    # Create a new array for dg
    dg = copy(g)
    
    # Call the in-place function
    fun_dg_nopop!(dg, g, params, T)
    
    # Return the result
    return dg
end

function fun_dg_nopop(g, params)
    # Create a new array for dg

    return fun_dg_nopop(g, params, 1.0)
end



end