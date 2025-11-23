# -----------------------
# GLOBAL CONSTANTS
# -----------------------
const EPS_F32 = Float32(1e-6)
const READOUT_THRESH = Float32(1e-4)

# -----------------------
# Utility math
# -----------------------

inverse_softplus(x) = log(exp(max(x, 1f-6)) - 1f0)

function renorm_to_true(p, t)
    sum_p = sum(p)
    sum_t = sum(t)
    if sum_t <= 0; return p; end
    if sum_p <= 1e-6
        return p .+ (sum_t / length(p))
    end
    return p .* (sum_t / sum_p)
end

# -----------------------
# Gradient Norm + Clipping
# -----------------------

function grad_norm(g)
    nm = 0f0
    if g === nothing
        return 0f0
    elseif g isa AbstractArray
        return sum(abs2, g)
    elseif g isa NamedTuple
        for k in keys(g)
            nm += grad_norm(g[k])
        end
    end
    return nm
end

function robust_update(p, g, lr; max_norm=1.0f0)
    g_sq = grad_norm(g)
    g_norm = sqrt(g_sq)
    scale = g_norm > max_norm ? max_norm / (g_norm + 1e-6f0) : 1f0
    return recursive_apply_update(p, g, lr*scale), g_norm, (scale < 1f0)
end

recursive_apply_update(p::AbstractArray, g::AbstractArray, lr) = p .- lr*g

function recursive_apply_update(p::NamedTuple, g::NamedTuple, lr)
    NamedTuple{keys(p)}(Tuple(
        haskey(g,k) && g[k] !== nothing ? recursive_apply_update(p[k], g[k], lr) : p[k]
        for k in keys(p)
    ))
end

recursive_apply_update(p, g, lr) = p
