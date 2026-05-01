module ApparentCompTools
include("ModelHelperFuns.jl")
using .ModelHelperFuns

using UnPack
export Jacobian_g

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
