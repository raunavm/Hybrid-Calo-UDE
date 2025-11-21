energy_penalty(p, t) = ((sum(p) - sum(t)) / (sum(t) + EPS_F32))^2

#TRAINING FORWARD & LOSS
function stageA_forward_raw(m, ps, st, x, tmpl)
    y, _ = Lux.apply(m, x, ps, st)
    E_inc_vec = x[5:5, :]
    y_pos = softplus.(y) .* E_inc_vec
    reshape(y_pos, size(tmpl))
end

# Modified to accept hyperparameters
function global_loss_A(p, stA, idxs, events, Xs, modelA, W_HIT, W_SPARSE)
    l = 0f0
    for i in idxs
        pred = stageA_forward_raw(modelA, p, stA, Xs[i], events[i])
        z_p = log1p.(pred.+EPS_F32); z_t = log1p.(events[i].+EPS_F32)
        w = map(e -> e > 0f0 ? W_HIT : 1f0, events[i])
        L_vox = sum(w.*(z_p.-z_t).^2)/sum(w)
        L_sparse = mean(abs.(pred))
        d_p=depth_profile(pred); d_t=depth_profile(events[i])
        L_depth = mean((d_p.-d_t).^2)/(mean(d_t.^2)+EPS_F32)
        r_p=radial_profile(pred); r_t=radial_profile(events[i])
        L_rad = mean((r_p.-r_t).^2)
        l += L_vox + W_SPARSE*L_sparse + 50f0*L_depth + 50f0*L_rad + 500f0*energy_penalty(pred, events[i])
    end
    return l/length(idxs)
end

# Modified to accept hyperparameters
function stageB_forward(m, ps, st, E_A, E_inc, RES_RANGE)
    geo = cat(repeat(X_COORDS,1,1,size(E_A,3)), repeat(Y_COORDS,1,1,size(E_A,3)), 
              repeat(sqrt.(R2_PLANE)./MAX_R,1,1,size(E_A,3)), reshape(repeat(S_COORDS,inner=H_GLOBAL*W_GLOBAL),H_GLOBAL,W_GLOBAL,size(E_A,3)), dims=4)
    val_scaled = E_A .* 1.0f0 
    inp = cat(val_scaled, log1p.(E_A.+EPS_F32), geo, fill(E_inc,H_GLOBAL,W_GLOBAL,L_GLOBAL), dims=4)
    x = reshape(inp, H_GLOBAL, W_GLOBAL, L_GLOBAL, 7, 1)
    y, _ = Lux.apply(m, x, ps, st)
    M = dropdims(y, dims=(4,5))
    scale = (E_inc / Float32(N_VOX_GLOBAL)) * RES_RANGE
    ΔE = scale .* tanh.(M)
    E_A .+ ΔE
end

# Modified to accept hyperparameters
function global_loss_B(p, stB, idxs, events, EAs, Eincs, modelB, W_VOX_B, W_RAD_B, RES_RANGE)
    l = 0f0
    for i in idxs
        raw = stageB_forward(modelB, p, stB, EAs[i], Eincs[i], RES_RANGE)
        final = renorm_to_true(max.(raw, 0f0), events[i])
        z_b = log1p.(final.+EPS_F32); z_t = log1p.(events[i].+EPS_F32)
        L_vox = mean(abs.(z_b .- z_t))
        d_p=depth_profile(final); d_t=depth_profile(events[i])
        L_depth = mean((d_p.-d_t).^2)/(mean(d_t.^2)+EPS_F32)
        r_p=radial_profile(final); r_t=radial_profile(events[i])
        L_rad = mean((r_p.-r_t).^2)
        l += W_VOX_B*L_vox + 50f0*L_depth + W_RAD_B*L_rad
    end
    return l/length(idxs)
end
