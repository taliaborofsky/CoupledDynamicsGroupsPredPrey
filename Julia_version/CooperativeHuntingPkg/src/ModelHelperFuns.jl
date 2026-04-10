module ModelHelperFuns
using UnPack

export scale_parameters, scale_parameters2
export  fun_H1, fun_H2, fun_alpha1, fun_alpha2, fun_f1, fun_f2, fun_W, fun_S_given_W
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

function fun_alpha2(x, parameters)
    #= retired =#
    @unpack α2_fun_type, α2_of_1, s2 = parameters

    if α2_fun_type == "constant"
        return α2_of_1
    else
        θ_2 = -log(1 / α2_of_1 - 1) / (1 - s2)
        return @. 1 / (1 + exp(-θ_2 * (x - s2)))
    end
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