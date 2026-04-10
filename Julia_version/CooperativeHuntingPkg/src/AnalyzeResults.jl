module AnalyzeResults

using UnPack
include("ModelHelperFuns.jl")
include("ModelFuns.jl")
using .ModelHelperFuns
using .ModelFuns
using LaTeXStrings


export get_p, get_meanx, get_prob_in_x, get_meanx_nosingle
export ylabel_dic, param_label_dic

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
    :scale => L"Scale, $\beta_1/\beta_2$"
)



function get_p(g::AbstractMatrix, x_max::Int)
    x = 1:x_max
    return sum(x .* g, dims = 1)  # Return a vector
end

function get_p(g::AbstractVector, x_max::Int)
    x = 1:x_max
    return sum(x .*g) # returns a scalar
end
function get_p(g::AbstractVector)
    x_max = length(g)
    get_p(g,x_max)
end

function get_meanx(g::AbstractMatrix, x_max::Int, p::AbstractMatrix)
    """
    Average group size any individual is in when `p` is a vector.
    g is a 2d matrix
    """
    x_vec = 1:x_max
    numerator = @. (x_vec^2) * g
    #mask = (p .> 1e-10) .& all(g .> 0, dims=1)
    mask = p.>1e-10
    numerator_sum = sum(numerator, dims=1)
    ans = copy(p)
    ans[mask] = @. numerator_sum[mask] / p[mask]
    @. ans[!mask] = 1.0
    return ans
end

function get_meanx_nosingle(g::AbstractVector, x_max::Int, P::Number)
    #=
    Expected group size an individual belongs to if it is in a group
    =#
    x_vec = 2:x_max
    numerator = @. x_vec^2 * g[2:end]
    if P < 1e-10 
        return 1.0
    else
        return sum(numerator) / (P - g[1])
    end
end

function get_meanx_nosingle(g, x_max)
    #= 
    get mean x excluding singletons, no P given
    =#
    p = get_p(g,x_max)
    mean_x_nosingle = get_meanx_nosingle(g, x_max, P)
    return mean_x_nosingle
end
function get_meanx(g::AbstractVector, x_max::Int, p::Number)
    #=
    Average group size any individual is in when `p` is a scalar.
    will add method for p a vector and g a matrix later
    =#
    x_vec = 1:x_max
    numerator = @. x_vec^2 * g
    if p < 1e-10 #&& all(g .< 1e-10)
        return 1.0
    else
        return sum(numerator) / p
    end
end

function get_meanx(g, x_max)
#=
    get_meanx without p given
=#
    p = get_p(g,x_max)
    mean_x = get_meanx(g, x_max, p)
    return mean_x
end

function get_prob_in_x(g::AbstractMatrix, p::AbstractVector, x_max)
    # find prob in group of size x, for g a matrix and p a vector
    # if length of p is n, then dimensions of g need to be n x x_max

        x=1:x_max
        num_in_gx = @. g * x'
        prob_in_x = @. num_in_gx / p
    end

function get_prob_in_x(g::AbstractMatrix, p::Number, x_max)
    # find prob in group of size x, for g a matrix and p a vector
        x=1:x_max
        num_in_gx = @. g * x'
        prob_in_x = @. num_in_gx / p
    end
    
function get_prob_in_x(g::AbstractVector, p::Number, x_max)
    x = 1:x_max
    num_in_gx = @. g * x
    prob_in_x = @. num_in_gx / p
end



end