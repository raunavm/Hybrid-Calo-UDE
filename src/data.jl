using HDF5, Statistics, Random, SparseArrays
import GraphNeuralNetworks

# -----------------------
# GLOBAL GEOMETRY
# -----------------------
global H_GLOBAL=0; global W_GLOBAL=0; global L_GLOBAL=0; global N_VOX_GLOBAL=0
global X_COORDS, Y_COORDS, R2_PLANE, S_COORDS, MAX_R, G_GLOBAL

const Z_DIM = 45; const A_DIM = 16; const R_DIM = 9
const N_EXPECT = Z_DIM * A_DIM * R_DIM

# -----------------------
# FILE PICKER (Dataset-2 priority)
# -----------------------
function pick_dataset_file(dir::String)
    preferred = [
        "dataset_2_1.hdf5", "dataset_2_2.hdf5",
        "dataset_2_photons_1.hdf5", "dataset_2_photons_2.hdf5"
    ]
    for fname in preferred
        path = joinpath(dir, fname)
        if isfile(path)
            println("[Info] Using Dataset-2 file: $fname")
            return path
        end
    end
    files = filter(f->endswith(f,".hdf5"), readdir(dir; join=true))
    if isempty(files)
        error("No .hdf5 found in $dir")
    end
    println("[WARN] Using fallback $(basename(files[1]))")
    return files[1]
end

# -----------------------
# LOAD DATA
# -----------------------
function load_data(DATA_DIR::String; max_events=1024)
    fpath = pick_dataset_file(DATA_DIR)
    println("[Info] Loading $fpath")

    showers, energies = h5open(fpath, "r") do f
        read(f["showers"]), read(f["incident_energies"])
    end

    # Geometry fix
    if size(showers,1) == N_EXPECT
        # OK
    elseif size(showers,2) == N_EXPECT
        showers = permutedims(showers, (2,1))
    else
        error("Bad geometry: size = $(size(showers)), expected 6480 in one dim.")
    end

    N_use = min(max_events, size(showers,2))
    E_inc = Float32.(vec(energies)[1:N_use])

    events = [
        permutedims(reshape(Float32.(showers[:,e]), (Z_DIM,A_DIM,R_DIM)), (2,3,1))
        for e in 1:N_use
    ]

    return events, E_inc
end

# -----------------------
# GEOMETRY INITIALIZATION
# -----------------------
function init_geometry!(example)
    global H_GLOBAL, W_GLOBAL, L_GLOBAL, N_VOX_GLOBAL
    global X_COORDS, Y_COORDS, R2_PLANE, S_COORDS, MAX_R
    global G_GLOBAL

    H_GLOBAL, W_GLOBAL, L_GLOBAL = size(example)
    N_VOX_GLOBAL = H_GLOBAL*W_GLOBAL*L_GLOBAL

    X_COORDS = zeros(Float32,H_GLOBAL,W_GLOBAL)
    Y_COORDS = zeros(Float32,H_GLOBAL,W_GLOBAL)
    R2_PLANE = zeros(Float32,H_GLOBAL,W_GLOBAL)

    for i in 1:H_GLOBAL
        ϕ = Float32(2π*(i-0.5)/H_GLOBAL)
        for j in 1:W_GLOBAL
            r = Float32((j-0.5)/W_GLOBAL)
            X_COORDS[i,j] = r*cos(ϕ)
            Y_COORDS[i,j] = r*sin(ϕ)
            R2_PLANE[i,j] = r^2
        end
    end

    MAX_R = sqrt(maximum(R2_PLANE))
    S_COORDS = Float32[(k-1)/max(L_GLOBAL-1,1) for k in 1:L_GLOBAL]

    # Build GNN graph
    rows = Int[]; cols = Int[]

    idx(x,y,z) = x + (y-1)*H_GLOBAL + (z-1)*H_GLOBAL*W_GLOBAL

    for z in 1:L_GLOBAL, y in 1:W_GLOBAL, x in 1:H_GLOBAL
        u = idx(x,y,z)
        push!(rows,u); push!(cols,u)
        left  = x==1 ? H_GLOBAL : x-1
        right = x==H_GLOBAL ? 1 : x+1
        push!(rows,u); push!(cols,idx(left,y,z))
        push!(rows,u); push!(cols,idx(right,y,z))
        if y>1 push!(rows,u); push!(cols,idx(x,y-1,z)); end
        if y<W_GLOBAL push!(rows,u); push!(cols,idx(x,y+1,z)); end
        if z>1 push!(rows,u); push!(cols,idx(x,y,z-1)); end
        if z<L_GLOBAL push!(rows,u); push!(cols,idx(x,y,z+1)); end
    end

    G_GLOBAL = GraphNeuralNetworks.GNNGraph(
        sparse(rows, cols, ones(Float32,length(rows)))
    )
end

# -----------------------
# INPUT FEATURE CONSTRUCTION
# -----------------------
function build_all_inputs(events, energies)
    H,W,L = size(events[1])
    N = H*W*L

    x_b = vec(repeat(X_COORDS,1,1,L))
    y_b = vec(repeat(Y_COORDS,1,1,L))
    r_b = vec(repeat(sqrt.(R2_PLANE),1,1,L)) ./ MAX_R
    s_b = repeat(S_COORDS, inner=H*W)

    out = Vector{Matrix{Float32}}(undef, length(events))

    for i in eachindex(events)
        E = energies[i]
        logE = log10(E+1f0)

        feats = zeros(Float32, 22, N)
        feats[1,:] .= x_b
        feats[2,:] .= y_b
        feats[3,:] .= r_b
        feats[4,:] .= s_b
        feats[5,:] .= E
        feats[6,:] .= logE

        row = 7
        for f in (1f0,2f0,4f0,8f0)
            w = 2f0*pi*f
            feats[row,:] .= sin.(w.*r_b); row+=1
            feats[row,:] .= cos.(w.*r_b); row+=1
            feats[row,:] .= sin.(w.*s_b); row+=1
            feats[row,:] .= cos.(w.*s_b); row+=1
        end

        out[i] = feats
    end

    return out
end
