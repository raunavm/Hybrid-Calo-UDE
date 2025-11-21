module ECalUDE

using LinearAlgebra, Statistics, Random, Printf, Dates, Logging
using HDF5, Zygote, Lux, Serialization, SparseArrays, Plots, NNlib
import GraphNeuralNetworks
using GNNLux: GCNConv
using Measures

#GLOBALS
# These are populated by init_geometry!
global H_GLOBAL=0; global W_GLOBAL=0; global L_GLOBAL=0; global N_VOX_GLOBAL=0
global X_COORDS, Y_COORDS, R2_PLANE, S_COORDS, MAX_R, G_GLOBAL
const EPS_F32 = Float32(1e-6)
const READOUT_THRESH = Float32(1e-4)

#INCLUDES
include("utils.jl")
include("data.jl")
include("models.jl")
include("training.jl")

#EXPORTS
export main_training_loop

#MAIN TRAINING LOOP
function main_training_loop(;
        # Config
        SEED = 1234,
        MAX_EVENTS = 1024,
        BATCH_SIZE = 16,
        STAGEA_STEPS = 2000,
        STAGEB_STEPS = 1000,
        # Hyperparameters
        W_HIT = Float32(100.0),
        W_SPARSE = Float32(0.05),
        W_VOX_B = Float32(5.0),
        W_RAD_B = Float32(10.0),
        RES_RANGE = Float32(2000.0),
        # Geometry
        Z_DIM = 45, A_DIM = 16, R_DIM = 9,
        # Paths & Toggles
        DATA_DIR = "data/",
        MODEL_SAVE_PATH = "results/final_paper_model.jls",
        DO_PLOTS = true
    )
    
    N_EXPECT = Z_DIM * A_DIM * R_DIM

    println("=== v32 STABLE MEAN RUN (Robust Config) ===")

    # 1. LOAD DATA
    events, Eincs = load_data(DATA_DIR, N_EXPECT, MAX_EVENTS, Z_DIM, A_DIM, R_DIM)
    scale = maximum([sum(E) for E in events])
    invs = 1.0f0 / max(scale, 1f-6)
    for i in eachindex(events); events[i] .*= invs; Eincs[i] *= invs; end
    init_geometry!(events[1])
    all_vox = reduce(vcat, [vec(e) for e in events])
    bias_init = inverse_softplus(Float32(mean(all_vox)/mean(Eincs)))
    Xs = build_all_inputs(events, Eincs)
    
    rng = MersenneTwister(SEED)
    idx = shuffle(rng, 1:length(events))
    n_tr = floor(Int, 0.8*length(idx)); t_idx, v_idx = idx[1:n_tr], idx[n_tr+1:end]

    # 2. TRAINING
    println(">>> Training Stage A (GNN)...")
    modelA = build_stageA_model(size(Xs[1],1), rng)
    psA, stA = Lux.setup(rng, modelA)
    psA = (conv1=psA.conv1, conv2=psA.conv2, conv3=psA.conv3, dense=(weight=psA.dense.weight, bias=fill(bias_init, size(psA.dense.bias))))
    
    best_loss_A = Inf32; best_psA = deepcopy(psA); LR_A = 1e-3
    for s in 1:STAGEA_STEPS
        batch = rand(rng, t_idx, BATCH_SIZE)
        # Pass hyperparams to loss
        l, back = Zygote.pullback(p -> global_loss_A(p, stA, batch, events, Xs, modelA, W_HIT, W_SPARSE), psA)
        if isnan(l) || isinf(l); continue; end
        gs = back(1f0)[1]
        psA, nm, clipped = robust_update(psA, gs, Float32(LR_A); max_norm=1.0f0)
        if s % 200 == 0; @printf("Step A %4d | Loss %.4f | |g| %.4f %s\n", s, l, nm, clipped ? "[Clipped]" : ""); end
        if s % 50 == 0
            v_l = global_loss_A(psA, stA, v_idx[1:min(end, 32)], events, Xs, modelA, W_HIT, W_SPARSE)
            if v_l < best_loss_A; best_loss_A = v_l; best_psA = deepcopy(psA); end
        end
    end
    EAs = [renorm_to_true(stageA_forward_raw(modelA, best_psA, stA, Xs[i], events[i]), events[i]) for i in 1:length(events)]
    
    println(">>> Training Stage B (CNN)...")
    modelB = build_stageB_model()
    psB, stB = Lux.setup(rng, modelB); best_loss_B = Inf32; best_psB = deepcopy(psB); LR_B = 1e-4
    for s in 1:STAGEB_STEPS
        batch = rand(rng, t_idx, BATCH_SIZE)
        # Pass hyperparams to loss
        l, back = Zygote.pullback(p -> global_loss_B(p, stB, batch, events, EAs, Eincs, modelB, W_VOX_B, W_RAD_B, RES_RANGE), psB)
        if isnan(l) || isinf(l); continue; end
        gs = back(1f0)[1]
        psB, nm, clipped = robust_update(psB, gs, Float32(LR_B); max_norm=1.0f0)
        if s % 200 == 0; @printf("Step B %4d | Loss %.4f | |g| %.4f %s\n", s, l, nm, clipped ? "[Clipped]" : ""); end
        if s % 50 == 0
            v_l = global_loss_B(psB, stB, v_idx[1:min(end, 32)], events, EAs, Eincs, modelB, W_VOX_B, W_RAD_B, RES_RANGE)
            if v_l < best_loss_B; best_loss_B = v_l; best_psB = deepcopy(psB); end
        end
    end

    println(">>> Saving Model...")
    serialize(MODEL_SAVE_PATH, (best_psA, best_psB))
    
    # 3. EVALUATION & PLOTS
    println(">>> Generating Final Plots...")
    if DO_PLOTS
        prof_rad_t = zeros(Float32, W_GLOBAL); prof_rad_p = zeros(Float32, W_GLOBAL)
        prof_dep_t = zeros(Float32, L_GLOBAL); prof_dep_p = zeros(Float32, L_GLOBAL)
        n_haze_pixels = 0; total_pixels = 0; energy_ratios = Float32[]

        println("\n--- EVENT DIAGNOSTICS (First 5) ---")
        @printf("%-6s | %-15s | %-10s\n", "ID", "Energy (T/P)", "Ratio")
        
        for i in v_idx
            rawA = stageA_forward_raw(modelA, best_psA, stA, Xs[i], events[i])
            EA = renorm_to_true(map(x->x<READOUT_THRESH ? 0f0 : x, rawA), events[i])
            rawB = stageB_forward(modelB, best_psB, stB, EA, Eincs[i], RES_RANGE)
            EB = renorm_to_true(map(x->x<READOUT_THRESH ? 0f0 : x, max.(rawB, 0f0)), events[i])
            
            # Metrics
            sum_t = sum(events[i]); sum_p = sum(EB)
            ratio = sum_p / (sum_t + 1f-6)
            push!(energy_ratios, ratio)
            
            if i <= 5
                @printf("%-6d | %-6.1f / %-6.1f | %.4f\n", i, sum_t, sum_p, ratio)
            end

            prof_rad_t .+= vec(mean(sum(events[i], dims=3), dims=1))
            prof_rad_p .+= vec(mean(sum(EB, dims=3), dims=1))
            prof_dep_t .+= vec(mean(sum(events[i], dims=(1,2)), dims=1))
            prof_dep_p .+= vec(mean(sum(EB, dims=(1,2)), dims=1))

            mask_true_zero = events[i] .== 0f0
            n_haze_pixels += count((EB .> READOUT_THRESH) .& mask_true_zero)
            total_pixels += length(EB)
        end
        
        prof_rad_t ./= length(v_idx); prof_rad_p ./= length(v_idx)
        prof_dep_t ./= length(v_idx); prof_dep_p ./= length(v_idx)
        
        # Plot 1: Radial + Ratio
        l = @layout [a; b{0.3h}]
        p1a = plot(prof_rad_t, label="Geant4", lw=3, color=:black, ylabel="Energy", title="Radial Profile", grid=true)
        plot!(p1a, prof_rad_p, label="Ours", lw=3, ls=:dash, color=:red)
        ratio_r = (prof_rad_p .+ 1f-6) ./ (prof_rad_t .+ 1f-6)
        p1b = plot(ratio_r, label="", lw=2, color=:black, ylabel="Ratio", xlabel="R", ylims=(0.5, 1.5), grid=true)
        hline!(p1b, [1.0], color=:grey, ls=:dash)
        p1 = plot(p1a, p1b, layout=l, size=(600,600)); display(p1) 
        
        # Plot 2: Depth + Ratio
        p2a = plot(prof_dep_t, label="Geant4", lw=3, color=:black, ylabel="Energy", title="Depth Profile", grid=true)
        plot!(p2a, prof_dep_p, label="Ours", lw=3, ls=:dash, color=:blue)
        ratio_d = (prof_dep_p .+ 1f-6) ./ (prof_dep_t .+ 1f-6)
        p2b = plot(ratio_d, label="", lw=2, color=:black, ylabel="Ratio", xlabel="Z", ylims=(0.5, 1.5), grid=true)
        hline!(p2b, [1.0], color=:grey, ls=:dash)
        p2 = plot(p2a, p2b, layout=l, size=(600,600)); display(p2)
        
        # Plot 3: Conservation
        p3 = histogram(energy_ratios, bins=40, label="Events", title="Conservation (Enforced)", xlabel="Pred/Truth", color=:green)
        vline!(p3, [1.0], color=:black, lw=2, label="Ideal")
        display(p3)

        # Plot 4: Heatmap
        idx = v_idx[1]
        rawA_vis = stageA_forward_raw(modelA, best_psA, stA, Xs[idx], events[idx])
        EA_vis = renorm_to_true(map(x->x<READOUT_THRESH ? 0f0 : x, rawA_vis), events[idx])
        rawB_vis = stageB_forward(modelB, best_psB, stB, EA_vis, Eincs[idx], RES_RANGE)
        EB_vis = renorm_to_true(map(x->x<READOUT_THRESH ? 0f0 : x, max.(rawB_vis, 0f0)), events[idx])
        
        slice_t = sum(events[idx], dims=3)[:,:,1]
        slice_p = sum(EB_vis, dims=3)[:,:,1]
        h1 = heatmap(log10.(slice_t .+ 1e-5), title="Geant4", c=:viridis)
        h2 = heatmap(log10.(slice_p .+ 1e-5), title="Ours", c=:viridis)
        display(plot(h1, h2, layout=(1,2), size=(800,400)))

        haze_percent = (n_haze_pixels / total_pixels) * 100
        println("\n=== FINAL METRICS ===")
        @printf("1. Haze Index: %.4f%%\n", haze_percent)
        @printf("2. Mean Energy Ratio: %.6f\n", mean(energy_ratios))
    end
end

end # end module ECalUDE
