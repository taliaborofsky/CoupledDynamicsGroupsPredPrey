module AnalyzeResults

export get_p, get_meanx, get_prob_in_x


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