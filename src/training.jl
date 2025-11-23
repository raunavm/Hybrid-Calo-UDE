using Lux, Zygote, NNlib
using ..ECalUDE: EPS_F32, READOUT_THRESH, renorm_to_true
using ..data: depth_profile, radial_profile, MAX_R, R2_PLANE, X_COORDS, Y_COORDS, S_COORDS
using ..utils: robust_update

# -----------------------
# Hyperparameters (v38 relax)
# -----------------------
const W_VOX_A    = Float32(10.0)
const W_SPARSE_A = Float32(0.1)
const W_VOX_B    = Float32(10.0)
const W_RAD_B    = Float32(10.0)
const W_DEPTH_B  = Float32(20.0)
const RES_RANGE  = Float32(1000.0)

# -----------------------
# Forward Passes
# -----------------------

function stageA_forward_raw(m, ps, st, x, tmpl)
    y,_ = Lux.apply(m, x, ps, st)
    E_inc_vec = x[5:5,:]
    y_pos = softplus.(y) .* E_inc_vec
    reshape(y_pos, size(tmpl))
end

function stageB_forward(modelB, ps, st, E_A, E_inc)
    H,W,L = size(E_A)

    geo = cat(
        repeat(X_COORDS,1,1,L),
        repeat(Y_COORDS,1,1,L),
        repeat(sqrt.(R2_PLANE)./MAX_R,1,1,L),
        reshape(repeat(S_COORDS, inner=H*W),H,W,L),
        dims=4
    )

    inp = cat(E_A, log1p.(E_A.+EPS_F32), geo,
              fill(E_inc,H,W,L), dims=4)

    x = reshape(inp, H,W,L,7,1)
    y,_ = Lux.apply(modelB, x, ps, st)
    M = dropdims(y, dims=(4,5))

    ΔE = (E_inc / Float32(H*W*L)) * RES_RANGE .* tanh.(M)
    return E_A .+ ΔE
end

# -----------------------
# Log Profile Loss
# -----------------------
function log_profile_loss(pred, true)
    log_p = log1p.(abs.(pred))
    log_t = log1p.(abs.(true))
    return mean(abs.(log_p .- log_t))
end

# -----------------------
# Loss Stage A
# -----------------------
function global_loss_A(p, stA, idxs, events, Xs, modelA)
    L = 0f0
    for i in idxs
        pred = stageA_forward_raw(modelA, p, stA, Xs[i], events[i])
        z_p = log1p.(pred.+EPS_F32)
        z_t = log1p.(events[i].+EPS_F32)

        # Log-Cosh voxel loss
        diff = z_p .- z_t
        L_vox = mean(log.(cosh.(W_VOX_A .* diff)))

        L_sparse = mean(abs.(pred))

        dp, dt = depth_profile(pred), depth_profile(events[i])
        L_depth = log_profile_loss(dp, dt)

        rp, rt = radial_profile(pred), radial_profile(events[i])
        L_rad = log_profile_loss(rp, rt)

        L += L_vox + W_SPARSE_A*L_sparse + W_DEPTH_B*L_depth + W_RAD_B*L_rad
    end
    return L/length(idxs)
end

# -----------------------
# Loss Stage B
# -----------------------
function global_loss_B(p, stB, idxs, events, EAs, Eincs, modelB)
    L = 0f0
    for i in idxs
        raw = stageB_forward(modelB, p, stB, EAs[i], Eincs[i])
        final = renorm_to_true(max.(raw,0f0), events[i])

        z_p = log1p.(final .+ EPS_F32)
        z_t = log1p.(events[i] .+ EPS_F32)

        L_vox = mean(log.(cosh.(W_VOX_B .* (z_p .- z_t))))

        dp, dt = depth_profile(final), depth_profile(events[i])
        L_depth = log_profile_loss(dp, dt)

        rp, rt = radial_profile(final), radial_profile(events[i])
        L_rad = log_profile_loss(rp, rt)

        L += L_vox + W_DEPTH_B*L_depth + W_RAD_B*L_rad
    end
    return L/length(idxs)
end
