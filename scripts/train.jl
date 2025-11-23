using Pkg
Pkg.activate("..")

using ECalUDE
using Random, Serialization, Printf

# --- CONFIG ---
const SEED          = 1234
const MAX_EVENTS    = 1024
const BATCH_SIZE    = 16
const STAGEA_STEPS  = 2000
const STAGEB_STEPS  = 2000

const DATA_DIR = joinpath(@__DIR__, "..", "data")

# -----------------------
# MAIN
# -----------------------
function main()
    println("=== Hybrid-Calo-UDE v38 TRAINING ===")

    # -------------------------
    # Load Data
    # -------------------------
    events, Eincs = load_data(DATA_DIR; max_events=MAX_EVENTS)

    # Robust scaling
    all_sum = [sum(E) for E in events]
    scale_val = quantile(all_sum, 0.99)
    println("[Info] Scale = $scale_val")

    invs = Float32(1/scale_val)
    for i in eachindex(events)
        events[i] .*= invs
        Eincs[i]  *= invs
    end

    # -------------------------
    # Build Geometry + Inputs
    # -------------------------
    init_geometry!(events[1])
    Xs = build_all_inputs(events, Eincs)

    # -------------------------
    # Data Split
    # -------------------------
    rng = MersenneTwister(SEED)
    idxs = shuffle(rng, 1:length(events))
    n_tr = floor(Int, 0.8*length(idxs))
    t_idx = idxs[1:n_tr]
    v_idx = idxs[n_tr+1:end]

    # -------------------------
    # Stage A Init
    # -------------------------
    modelA = build_stageA_model(size(Xs[1],1), rng)
    psA, stA = Lux.setup(rng, modelA)

    # Bias init
    all_vox = reduce(vcat, [vec(e) for e in events])
    bias_init = inverse_softplus(Float32(mean(all_vox)/mean(Eincs)) * 2f0)
    psA = (conv1=psA.conv1, conv2=psA.conv2, conv3=psA.conv3,
           dense=(weight=psA.dense.weight, bias=fill(bias_init,size(psA.dense.bias))))

    # -------------------------
    # Train Stage A
    # -------------------------
    println(">>> Training Stage A...")
    best_loss_A = Inf32
    best_psA = deepcopy(psA)
    LR_A = 1e-3

    for step in 1:STAGEA_STEPS
        batch = rand(rng, t_idx, BATCH_SIZE)
        l,back = Zygote.pullback(p->global_loss_A(p,stA,batch,events,Xs,modelA), psA)
        if !isfinite(l); continue; end
        gs = back(1f0)[1]

        psA, gnorm, clipped = robust_update(psA, gs, Float32(LR_A))

        if step % 200 == 0
            @printf("A[%4d] loss=%.4f |g|=%.4f %s\n",
                step, l, gnorm, clipped ? "[CLIP]" : "")
        end

        if step % 50 == 0
            val = global_loss_A(psA, stA, v_idx[1:min(end,32)], events, Xs, modelA)
            if val < best_loss_A
                best_loss_A = val
                best_psA = deepcopy(psA)
            end
        end
    end

    # Stage A predictions
    EAs = [
        renorm_to_true(stageA_forward_raw(modelA,best_psA,stA,Xs[i],events[i]), events[i])
        for i in 1:length(events)
    ]

    # -------------------------
    # Stage B Init
    # -------------------------
    modelB = build_stageB_model()
    psB, stB = Lux.setup(rng, modelB)

    # -------------------------
    # Train Stage B
    # -------------------------
    println(">>> Training Stage B...")
    best_loss_B = Inf32
    best_psB = deepcopy(psB)
    LR_B = 1e-4

    for step in 1:STAGEB_STEPS
        batch = rand(rng, t_idx, BATCH_SIZE)
        l,back = Zygote.pullback(p->global_loss_B(p,stB,batch,events,EAs,Eincs,modelB), psB)
        if !isfinite(l); continue; end
        gs = back(1f0)[1]

        psB, gnorm, clipped = robust_update(psB, gs, Float32(LR_B))

        if step % 200 == 0
            @printf("B[%4d] loss=%.4f |g|=%.4f %s\n",
                step, l, gnorm, clipped ? "[CLIP]" : "")
        end

        if step % 50 == 0
            val = global_loss_B(psB, stB, v_idx[1:min(end,32)], events, EAs, Eincs, modelB)
            if val < best_loss_B
                best_loss_B = val
                best_psB = deepcopy(psB)
            end
        end
    end

    # -------------------------
    # Save Model
    # -------------------------
    serialize(joinpath(@__DIR__,"..","results","final_model_v38.jls"),
              (best_psA, best_psB))

    println(">>> DONE.")
end

main()
