#PHYSICS HELPERS
depth_profile(E) = vec(sum(E, dims=(1,2)))
function radial_profile(E)
    H, W, L = size(E)
    buf = Zygote.Buffer(zeros(Float32, L))
    for k in 1:L
        Ek = @view E[:,:,k]; Et = sum(Ek)
        if Et <= 0; buf[k] = 0f0
        else; mean_r2 = sum(R2_PLANE .* Ek) / Et; buf[k] = sqrt(max(mean_r2, 0f0)) / MAX_R; end
    end
    copy(buf)
end

#DATA LOADING
function pick_dataset_file(dir::String)
    preferred = ["dataset_2_1.hdf5", "dataset_2_2.hdf5", "dataset_2_photons_1.hdf5"]
    for fname in preferred
        path = joinpath(dir, fname)
        if isfile(path)
            println("[Info] Found preferred dataset: $path")
            return path 
        end
    end
    files = filter(f -> endswith(f, ".hdf5") || endswith(f, ".h5"), readdir(dir; join=true))
    isempty(files) && error("No .hdf5 files found in $dir")
    println("[Warn] No dataset_2 found. Using fallback: $(files[1])")
    return files[1]
end

# Modified to accept parameters
function load_data(data_dir::String, n_expect::Int, max_events::Int, Z_DIM::Int, A_DIM::Int, R_DIM::Int)
    fpath = pick_dataset_file(data_dir)
    println("[Info] Loading HDF5...")
    showers, energies = h5open(fpath, "r") do f
        (read(f["showers"]), read(f["incident_energies"]))
    end
    
    # Geometry Check
    n_cells = (ndims(showers)==2) ? size(showers, 1) : size(showers, 2)
    if n_cells != n_expect
        if size(showers, 2) == n_expect
             showers = permutedims(showers, (2,1))
        else
             error("CRITICAL ERROR: Dataset geometry mismatch. Expected $n_expect voxels.")
        end
    end

    N_events = size(showers, 2)
    N_use = min(max_events, N_events)
    E_inc = Float32.(vec(energies)[1:N_use])
    events = Vector{Array{Float32,3}}(undef, N_use)
    
    for e in 1:N_use
        flat = Float32.(showers[:,e])
        X_zar = reshape(flat, (Z_DIM, A_DIM, R_DIM))
        events[e] = permutedims(X_zar, (2,3,1))
    end
    return events, E_inc
end

function init_geometry!(example)
    global H_GLOBAL, W_GLOBAL, L_GLOBAL, N_VOX_GLOBAL, X_COORDS, Y_COORDS, R2_PLANE, S_COORDS, MAX_R, G_GLOBAL
    H_GLOBAL, W_GLOBAL, L_GLOBAL = size(example)
    N_VOX_GLOBAL = H_GLOBAL*W_GLOBAL*L_GLOBAL
    println("[Info] Initializing Geometry: H=$H_GLOBAL, W=$W_GLOBAL, L=$L_GLOBAL")
    
    X_COORDS = zeros(Float32, H_GLOBAL, W_GLOBAL); Y_COORDS = zeros(Float32, H_GLOBAL, W_GLOBAL); R2_PLANE = zeros(Float32, H_GLOBAL, W_GLOBAL)
    for i in 1:H_GLOBAL
        ϕ = Float32(2π * (i - 0.5) / H_GLOBAL)
        for j in 1:W_GLOBAL
            r = Float32((j - 0.5) / W_GLOBAL); X_COORDS[i,j]=r*cos(ϕ); Y_COORDS[i,j]=r*sin(ϕ); R2_PLANE[i,j]=r^2
        end
    end
    MAX_R = sqrt(maximum(R2_PLANE))
    S_COORDS = Float32[ (k-1)/max(L_GLOBAL-1, 1) for k in 1:L_GLOBAL ]
    
    rows, cols = Int[], Int[]
    idx(x,y,z) = x + (y-1)*H_GLOBAL + (z-1)*H_GLOBAL*W_GLOBAL
    for z in 1:L_GLOBAL, y in 1:W_GLOBAL, x in 1:H_GLOBAL
        u=idx(x,y,z); push!(rows,u); push!(cols,u)
        l=(x==1) ? H_GLOBAL : x-1; r=(x==H_GLOBAL) ? 1 : x+1
        push!(rows,u); push!(cols,idx(l,y,z)); push!(rows,u); push!(cols,idx(r,y,z))
        if y>1 push!(rows,u); push!(cols,idx(x,y-1,z)) end
        if y<W_GLOBAL push!(rows,u); push!(cols,idx(x,y+1,z)) end
        if z>1 push!(rows,u); push!(cols,idx(x,y,z-1)) end
        if z<L_GLOBAL push!(rows,u); push!(cols,idx(x,y,z+1)) end
    end
    G_GLOBAL = GraphNeuralNetworks.GNNGraph(sparse(rows, cols, ones(Float32, length(rows))))
    println("[Info] Graph Initialized.")
end

function build_all_inputs(events, energies)
    H,W,L = size(events[1]); N = H*W*L
    x_b = vec(repeat(X_COORDS,1,1,L)); y_b = vec(repeat(Y_COORDS,1,1,L))
    r_b = vec(repeat(sqrt.(R2_PLANE),1,1,L))./MAX_R; s_b = repeat(S_COORDS, inner=H*W)
    X_list = Vector{Matrix{Float32}}(undef, length(events))
    for i in eachindex(events)
        E = energies[i]; feats = zeros(Float32, 22, N)
        feats[1,:] .= x_b; feats[2,:] .= y_b; feats[3,:] .= r_b; feats[4,:] .= s_b
        feats[5,:] .= E;   feats[6,:] .= log1p(E)
        row=7
        for f in [1f0, 2f0, 4f0, 8f0]
            w = 2f0*Float32(π)*f
            feats[row,:] .= sin.(w.*r_b); row+=1; feats[row,:] .= cos.(w.*r_b); row+=1
            feats[row,:] .= sin.(w.*s_b); row+=1; feats[row,:] .= cos.(w.*s_b); row+=1
        end
        X_list[i] = feats
    end
    X_list
end
