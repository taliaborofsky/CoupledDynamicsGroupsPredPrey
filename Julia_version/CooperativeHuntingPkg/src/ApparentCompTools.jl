module ApparentCompTools
include("ModelHelperFuns.jl")
include("ModelFuns.jl")
include("AnalyzeResults.jl")
include("MyBifTools.jl")
using .ModelHelperFuns
using .ModelFuns
using .AnalyzeResults
using .MyBifTools

using UnPack
using LinearAlgebra
export Jacobian, Jacobian_g
export get_∂N2_∂N1, get_∂N1_∂N2

function Jacobian(N1, N2, g, params_use)

    @unpack x_max, η1, η2, β1, β2, d, Tg = params_use
    xvec = 1:x_max
    δ = 1 - η1 - η2
    f_1 = fun_f1(xvec, N1, N2, params_use)
    f_2 = fun_f2(xvec,N1,N2,params_use)
    partial_f_1 = fun_grad_func_response(1, xvec, N1, N2, params_use)
    partial_f_2 = fun_grad_func_response(2, xvec, N1, N2, params_use)

    # N1 row
    ∂N1 = [η1 *(1 - 2*N1) - dot(g,partial_f_1[1]),
                        -dot(g, partial_f_1[2]), -f_1...]
    # N2 row
    ∂N2 = [-dot(g, partial_f_2[1]), 
                        η2*(1 - 2*N2) - dot(g, partial_f_2[2]), -f_2...]

    ## for group section
    # get partial W wrt N1
    ∂W = [(β1 .* partial_f_1[i] .+ β2 .* partial_f_2[i]) ./ xvec for i in 1:2]
    W = (β1 .* f_1 + β2 .* f_2) ./ xvec

    # need S(x,1)

    S_x_1 = fun_S_given_W(W,W[1], params_use)
    S_1_x = 1 .- S_x_1
    # partial of S(x,1) wrt N1, N2
    ∂S_x_1 = [ d .* S_x_1 .* (1 .- S_x_1) .* (∂W[i] .- ∂W[i][1]) for i in 1:2] 
    ∂S_1_x = - ∂S_x_1
    # now construct g matrix. for now this works for xmax > 2
    J_g_mat = zeros(x_max, x_max+2)
    δ = 1 - η1 - η2
    for x in xvec
        if x == 1
            Q1_Ni_g = [(2*g[2]*∂S_1_x[i][2] 
            + sum(xvec[2:end] .* g[2:end] .* ∂S_1_x[i][2:end] .- g[1].*g[1:end-1] .* ∂S_x_1[i][2:end]))/Tg
            for i in 1:2]
            Q1_Ni_pop = [x_max*g[x_max]*∂W[i][end] - g[1]*∂W[i][1]
                for i in 1:2]
            Q1_Ni = Q1_Ni_g .+ Q1_Ni_pop
            ∂Q1_g1 = (-2*g[1]*S_x_1[2] - dot(g[2:end-1], S_x_1[3:end]))/Tg  - W[1] - δ
            ∂Q1_g2 = (4*S_1_x[2] - g[1] * S_x_1[3])/Tg + 2*δ
            ∂Q1_gx = [(y * S_1_x[y] - g[1] * S_x_1[y+1])/Tg for y in 3:x_max - 1]
            ∂Q1_gxmax = x_max*S_1_x[end]/Tg + x_max*W[end]
            J_g_mat[x,:] = [Q1_Ni..., ∂Q1_g1, ∂Q1_g2, ∂Q1_gx..., ∂Q1_gxmax]
        elseif x == 2
            ∂Q2_Ni_g = [
                (-∂S_1_x[i][2] *(2*g[2] + g[1]^2/2) + ∂S_1_x[i][3] * (3 * g[3] + g[1] * g[2]))/Tg 
                for i in 1:2]
            ∂Q2_Ni_pop =  [g[1]*∂W[i][1] - 2*g[2]*∂W[i][2] 
                for i in 1:2]
            ∂Q2_Ni = ∂Q2_Ni_g .+ ∂Q2_Ni_pop
            ∂Q2_g1 = (g[1]*S_x_1[2] - g[2] * S_x_1[3])/Tg + W[1]
            ∂Q2_g2 = -(2*S_1_x[2] + g[1]*S_x_1[3])/Tg - 2*W[2] - 2*δ
            ∂Q2_g3 = 3*S_1_x[3]/Tg + 3 * δ
            J_g_mat[x,:] = [∂Q2_Ni..., ∂Q2_g1, ∂Q2_g2, ∂Q2_g3, zeros(x_max-3)...]
        elseif x == x_max
            ∂Qxm_Ni_g = [
                -∂S_1_x[i][x]*(x*g[x] + g[1]*g[x-1])/Tg
            for i in 1:2]
            ∂Qxm_Ni_pop = [ 
                 (x-1)*g[x-1]*∂W[i][x-1] 
                for i in 1:2]   
            ∂Qxm_Ni = ∂Qxm_Ni_g .+ ∂Qxm_Ni_pop
            ∂Qxm_g = zeros(x_max)
            ∂Qxm_g[1] = (g[x-1]*S_x_1[x])/Tg
            ∂Qxm_g[x-1] = g[1]*S_x_1[x]/Tg + (x-1)*W[x-1]
            ∂Qxm_g[x] = -( x*S_1_x[x])/Tg - x*δ
            J_g_mat[x,:] = [∂Qxm_Ni..., ∂Qxm_g...]
        else
            ∂Qx_Ni_group = [
                (∂S_1_x[i][x+1] *((x+1)*g[x+1] + g[1]*g[x]) - ∂S_1_x[i][x] * (x * g[x] + g[1]*g[x-1]))/Tg 
                for i in 1:2]
            ∂Qx_Ni_pop = [(x-1)*g[x-1]*∂W[i][x-1] - x*g[x]*∂W[i][x] 
                    for i in 1:2]
            ∂Qx_Ni = ∂Qx_Ni_group .+ ∂Qx_Ni_pop
            ∂Qx_g = zeros(x_max)
            ∂Qx_g[1] = (g[x-1]*S_x_1[x] - g[x]*S_x_1[x+1])/Tg
            ∂Qx_g[x-1] = g[1]*S_x_1[x]/Tg + (x-1)*W[x-1]
            ∂Qx_g[x] = -( x*S_1_x[x] + g[1]*S_x_1[x+1] )/Tg - x* ( W[x] + δ )
            ∂Qx_g[x+1] = (x+1)*S_1_x[x+1]/Tg + (x+1)*δ
            J_g_mat[x,:] = [∂Qx_Ni..., ∂Qx_g...]
        end
    end
    J_analytical = vcat(∂N1', ∂N2', J_g_mat)
end

function fun_grad_func_response(i, x, N1, N2, params)
    """
    The gradient of the (scaled) functional response on prey i wrt N1, N2.
    Returns a 2x(length(x)) matrix where rows correspond to N1 and N2.
    """
    alpha1 = fun_alpha1(x, params)
    alpha2 = fun_alpha2(x, params)
    H1 = fun_H1(x, params)
    H2 = fun_H2(x, params)
    denom = (1 .+ alpha1 .* H1 .* N1 .+ alpha2 .* N2 .* H2).^2

    @unpack A1, A2 = params
    if i == 1 # partial of f1 wrt N1 and N2
        return [(A1 .* alpha1 .* (1 .+ alpha2 .* H2 .* N2)) ./ denom, # wrt N1
                (-A1 .* alpha1 .* alpha2 .* H2 .* N1) ./ denom] # wrt N2
    elseif i == 2 #partial of f2 wrt N1 and N2
        return [ (-A2 .* alpha1 .* alpha2 .* H1 .* N2) ./denom , # wrt N1
                (A2 .* alpha2 .* (1 .+ alpha1 .* H1 .* N1)) ./ denom] # wrt N2
    else
        error("Invalid value for i. Must be 1 or 2.")
    end
end

function get_∂N2_∂N1(Jac_full)
    """
    This obtains ∂N2 / ∂N1.
    To find apparent competition, we solve the equation:
        -(∂U₂, Q₁, Q₂, ... Qₓₘ derived all with respect to N1)
        = J_apparent * (∂N₂, g(1), g(2), ..., g(xₘ) derived all with respect to N₁),
    where J_apparent is the full Jacobian of the system without the partial derivatives wrt N₁.
    i.e., without the first row and column (note U₂ is the right side of dN₂/dT, Qᵢ is the right side of dgᵢ/dT).

    There is apparent competition if ∂N₂ / ∂N₁ is negative.

    Inputs:
    - Jac_full: Jacobian at stable coexistence equilibrium

    Returns:
    - The partial derivative of N₂ with respect to N₁ (Float64)
    """
    # Trim the first row and column
    Jac_apparent = Jac_full[2:end, 2:end]

    # Get the left side of the equation
    left_side_eqn = -Jac_full[2:end, 1] # 1st column except for first row

    # Solve for partial derivatives
    pdv_N1 = Jac_apparent \ left_side_eqn

    # Extract the apparent competition value
    apparent_comp_on_2 = pdv_N1[1]

    return apparent_comp_on_2
end

function get_∂N1_∂N2(Jac_full)
    """
    This obtains ∂N1 / ∂N2.
    To find apparent competition, we solve the equation:
        -(∂U₁, Q₁, Q₂, ... Qₓₘ all with respect to N₂)
        = J_apparent * (∂N₁, g(1), g(2), ..., g(xₘ) all with respect to N₂),
    where J_apparent is the full Jacobian of the system without the partial derivatives wrt N₂.
    i.e., without the 2nd row and column (note Uᵢ is the right side of dNᵢ/dT, Qᵢ is the right side of dgᵢ/dT).

    There is apparent competition on 2 if ∂N₁ / ∂N₂ is negative.

    Inputs:
    - Jac_full: Jacobian at stable coexistence equilibrium

    Returns:
    - The partial derivative of N₂ with respect to N₁ (Float64)
    """
    # Trim the second row and column
    Jac_apparent = Jac_full[1:end .!=2, 1:end .!=2]

    # Get the left side of the equation... the 2nd col of the jacobian without the 2nd row
    left_side_eqn = - Jac_full[1:end .!= 2, 2]

    # Solve for partial derivatives
    pdv_N2 = Jac_apparent \ left_side_eqn

    # Extract the apparent competition value
    apparent_comp_on_1 = pdv_N2[1]

    return apparent_comp_on_1
end

function Jacobian_g(N1, N2, g, params_use)

    @unpack x_max, β1, β2, d = params_use
    xvec = 1:x_max
    f_1 = fun_f1(xvec, N1, N2, params_use)
    f_2 = fun_f2(xvec, N1, N2,params_use)
    Tg = 1

    ## for group section
    W = (β1 .* f_1 + β2 .* f_2) ./ xvec

    # need S(x,1)

    S_x_1 = fun_S_given_W(W,W[1], params_use)
    S_1_x = 1 .- S_x_1
    
    # now construct g matrix. for now this works for xmax > 2
    J_g_mat = zeros(x_max-1, x_max-1)
    for x in xvec[2:end]
        if x == 2
            ∂Q2_g2 = -(2*S_1_x[2] + g[1]*S_x_1[3])/Tg
            ∂Q2_g3 = 3*S_1_x[3]/Tg 
            J_g_mat[x-1,:] = [∂Q2_g2, ∂Q2_g3, zeros(x_max-3)...]
        elseif x == x_max
            ∂Qxm_g = zeros(x_max-1)
            ∂Qxm_g[x-2] = g[1]*S_x_1[x]/Tg  # wrt x - 1
            ∂Qxm_g[x-1] = -( x*S_1_x[x])/Tg # wrt x
            J_g_mat[x-1,:] = ∂Qxm_g
        else # 2 < x < xm
            ∂Qx_g = zeros(x_max-1)
            ∂Qx_g[x-2] = g[1]*S_x_1[x]/Tg # wrt x - 1
            ∂Qx_g[x-1] = -( x*S_1_x[x] + g[1]*S_x_1[x+1] )/Tg # wrt x
            ∂Qx_g[x] = (x+1)*S_1_x[x+1]/Tg #wrt x + 1
            J_g_mat[x-1,:] = ∂Qx_g
        end
    end
    return J_g_mat
end


end