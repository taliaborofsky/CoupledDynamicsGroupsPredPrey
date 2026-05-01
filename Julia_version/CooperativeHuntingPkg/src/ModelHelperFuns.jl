module ModelHelperFuns
using UnPack

export scale_parameters, scale_parameters2
export  fun_H1, fun_H2, fun_alpha1, fun_f1, fun_f2, fun_W
export fun_W_orig, fun_S_given_W, scale_parameters_orig
#=
all functions designed to broadcast over elements of x
except for fun_S_given_W, which broadcasts over Wx, Wy
=#


# using multiple dispatch for scale_parameters
function scale_parameters(parameters::NamedTuple)
    @unpack scale, β2, η2, A1, A2, H2a, H2b, H1a = parameters
    return merge(parameters,(β1 = scale*β2, 
                                η1 = η2/scale,
                                H1b = scale * (A1/A2) * (H2a + H2b) 
                                            - H1a))
end

function scale_parameters(parameters::Dict)
    @unpack scale, β2, η2, A1, A2, H2a, H2b, H1a = parameters
    parameters[:β1] = scale*β2
    parameters[:η1] = η2/scale
    parameters[:H1b] = scale * (A1/A2) * (H2a + H2b) - H1a
    return parameters
end

# using multiple dispatch for scale_parameters
"""
scale_parameters2
given H1b and H2(1), find H1a
"""
function scale_parameters2(parameters::NamedTuple)
    @unpack scale, β2, η2, A1, A2, H2a, H2b, H1b = parameters
    return merge(parameters,(β1 = scale*β2, 
                                η1 = η2/scale,
                                H1a = scale * (A1/A2) * (H2a + H2b) 
                                            - H1b))
end
"""
scale_parameters_orig
Allometric scaling in original (dimensional) parameter space.
Given mass_ratio = b1/b2 = (h1a+h1b) / (h2a+h2b), derive b1 and h1b.
No non-dim corrections (no A_i, k_i factors).
"""
function scale_parameters_orig(parameters::NamedTuple)
    @unpack mass_ratio, b2, h1a, h2a, h2b = parameters
    return merge(parameters, (
        b1  = b2 * mass_ratio,
        h1b = (h2a + h2b) * mass_ratio - h1a
    ))
end
function scale_parameters_orig(parameters::Dict)
    @unpack mass_ratio, b2, h1a, h2a, h2b = parameters
    parameters[:b1]  = b2 * mass_ratio
    parameters[:h1b] = (h2a + h2b) * mass_ratio - h1a
    return parameters
end

"""
fun_W_orig
Per-capita fecundity W from eq. fecundity_dim using original (dimensional) parameters:
    W(x,M1,M2) = (b1*f1 + b2*f2)/x
where f_i is the Type-II functional response (eq. fun_response):
    f_i = a_i*α_i(x)*M_i / (1 + a1*α1(x)*h1(x)*M1 + a2*α2*h2(x)*M2)
and h_i(x) = h_ia + h_ib/x.
Params must contain: a1, a2, b1, b2, h1a, h1b, h2a, h2b, α1_of_1, s1, α2_of_1.
"""
function fun_W_orig(x, M1, M2, params)
    @unpack a1, a2, b1, b2, h1a, h1b, h2a, h2b, α2_of_1 = params
    h1x  = @. h1a + h1b / x
    h2x  = @. h2a + h2b / x
    α1x  = fun_alpha1(x, params)
    α2   = α2_of_1
    denom = @. 1 + a1 * α1x * h1x * M1 + a2 * α2 * h2x * M2
    f1 = @. a1 * α1x * M1 / denom
    f2 = @. a2 * α2  * M2 / denom
    return @. (b1 * f1 + b2 * f2) / x
end

function scale_parameters2(parameters::Dict)
    @unpack scale, β2, η2, A1, A2, H2a, H2b, H1b = parameters
    parameters[:β1] = scale*β2
    parameters[:η1] = η2/scale
    parameters[:H1a] = scale * (A1/A2) * (H2a + H2b) - H1b
    return parameters
end

function fun_H1(x, parameters)

    @unpack H1a, H1b = parameters 

    @. H1a + H1b / x
end

function fun_H2(x, parameters)
    @unpack H2a, H2b = parameters 
    return @. H2a + H2b / x
end

function fun_alpha1(x, parameters)
    @unpack α1_of_1, s1 = parameters
    
    θ_1 = -log(1/α1_of_1 - 1) / (1 - s1)
    return @. 1 / (1 + exp(-θ_1 * (x - s1)))
end


fun_f1(x, N1, N2, parameters) = fun_response_non_dim(x, N1, N2, 1, parameters)
fun_f2(x, N1, N2, parameters) = fun_response_non_dim(x, N1, N2, 2, parameters)

function fun_response_non_dim(x, N1, N2, index, parameters)
    #=
    functional response to prey type index
    can handle x, N1, or N2 as vectors
    =#
    @unpack A1, A2, α2_of_1 = parameters

    H1 = fun_H1(x, parameters)
    H2 = fun_H2(x, parameters)
    α1 = fun_alpha1(x, parameters)
    α2 = α2_of_1 #fun_alpha2(x, parameters)

    if index == 1
        numerator = @. A1 * α1 * N1
    elseif index == 2  
        numerator = @. A2 * α2 * N2
    else
        error("Invalid index: must be 1 or 2")
    end

    denominator = @. 1 + α1 * H1 * N1 + α2 * H2 * N2

    return @. numerator / denominator
end

function fun_W(x, N1, N2, parameters)
    #= 
    per capita fecundity from prey
    x can be vector
    =#
    @unpack β1, β2 = parameters
    f1 = fun_f1(x,N1,N2,parameters)
    f2 = fun_f2(x,N1,N2,parameters)

    # this is what's being returned
    W = @. (β1*f1 + β2*f2)/x
end

function fun_S_given_W(Wx,Wy, parameters)
    @unpack d = parameters
    S = @. 1 /(1 + exp(-d*(Wx - Wy)))
end

end