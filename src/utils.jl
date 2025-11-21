#UTILS
inverse_softplus(x) = log(exp(max(x, 1f-6)) - 1f0)
renorm_to_true(p, t) = (sum(p)<=0 || sum(t)<=0) ? p : p .* (sum(t)/sum(p))

#GRADIENT CLIPPING
function grad_norm(g)
    nm = 0f0
    if g === nothing; return 0f0; end
    if g isa AbstractArray; return sum(abs2, g); end 
    if g isa NamedTuple
        for k in keys(g); nm += grad_norm(g[k]); end
    end
    return nm
end
function robust_update(p, g, lr; max_norm=1.0f0)
    g_sq_sum = grad_norm(g)
    g_norm = sqrt(g_sq_sum)
    scale = 1.0f0
    if g_norm > max_norm; scale = max_norm / (g_norm + Float32(1e-6)); end
    eff_lr = lr * scale
    return recursive_apply_update(p, g, eff_lr), g_norm, (scale < 1.0f0)
end
function recursive_apply_update(p::AbstractArray, g::AbstractArray, lr::Float32)
    return p .- lr .* g
end
function recursive_apply_update(p::NamedTuple, g::NamedTuple, lr::Float32)
    new_vals = []
    for k in keys(p)
        if haskey(g, k) && g[k] !== nothing; push!(new_vals, recursive_apply_update(p[k], g[k], lr)); else; push!(new_vals, p[k]); end
    end
    return NamedTuple{keys(p)}(Tuple(new_vals))
end
recursive_apply_update(p, g, lr) = p
