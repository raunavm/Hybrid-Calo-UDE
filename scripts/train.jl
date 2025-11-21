# To run this script:
# 1. Open your terminal
# 2. cd /path/to/your/Hybrid-Calo-UDE
# 3. Run: julia --project scripts/train.jl

using Pkg
Pkg.activate(".") # Ensures the project's environment is used

#PLOTTING SETUP
# This must be done *before* loading the main module
DO_PLOTS = true
try
    if get(ENV, "GKSwstype", "") == ""
        ENV["GKSwstype"] = "100"
    end
    @eval using Plots
    try; using Measures; catch; end 
catch e
    @warn "Plots.jl not available; disabling plots" err = e
    global DO_PLOTS = false
end

# Now, load our module
# *** IMPORTANT: This must match your module name in `src/ECalUDE.jl` ***
# If you rename the module inside that file, you must rename it here.
using ECalUDE 

#RUN TRAINING
ECalUDE.main_training_loop(
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
    DATA_DIR = "data/", # Points to your data folder
    MODEL_SAVE_PATH = "results/final_paper_model.jls", # Points to your results folder
    DO_PLOTS = DO_PLOTS
)
