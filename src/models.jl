using Lux, NNlib
using ..ECalUDE: physics_pad_3d
using GNNLux: GCNConv

# -----------------------
# Stage A: GNN
# -----------------------

struct StageAGCN{C1,C2,C3,D} <: Lux.AbstractLuxLayer
    conv1::C1
    conv2::C2
    conv3::C3
    dense::D
end

function (m::StageAGCN)(x, ps, st)
    g = ECalUDE.G_GLOBAL

    x1,_ = m.conv1(g,x,ps.conv1,st.conv1)
    x2,_ = m.conv2(g,tanh.(x1),ps.conv2,st.conv2)
    x3,_ = m.conv3(g,tanh.(x2),ps.conv3,st.conv3)
    y,st4 = m.dense(tanh.(x3),ps.dense,st.dense)

    return y, (conv1=st.conv1, conv2=st.conv2, conv3=st.conv3, dense=st4)
end

Lux.initialparameters(rng::AbstractRNG, m::StageAGCN) = (
    conv1 = Lux.initialparameters(rng,m.conv1),
    conv2 = Lux.initialparameters(rng,m.conv2),
    conv3 = Lux.initialparameters(rng,m.conv3),
    dense = Lux.initialparameters(rng,m.dense),
)

Lux.initialstates(rng::AbstractRNG, m::StageAGCN) = (
    conv1 = Lux.initialstates(rng,m.conv1),
    conv2 = Lux.initialstates(rng,m.conv2),
    conv3 = Lux.initialstates(rng,m.conv3),
    dense = Lux.initialstates(rng,m.dense),
)

build_stageA_model(in_feats, rng) = StageAGCN(
    GCNConv(in_feats=>64),
    GCNConv(64=>64),
    GCNConv(64=>64),
    Lux.Dense(64=>1)
)

# -----------------------
# Stage B: 3D CNN Residual
# -----------------------
function physics_pad_3d(x)
    x_h = cat(x[end:end,:,:,:,:], x, x[1:1,:,:,:,:]; dims=1)
    x_hw = cat(x_h[:,1:1,:,:,:], x_h, x_h[:,end:end,:,:,:]; dims=2)
    x_hwl = cat(x_hw[:,:,1:1,:,:], x_hw, x_hw[:,:,end:end,:,:]; dims=3)
    return x_hwl
end

function build_stageB_model()
    Lux.Chain(
        Lux.WrappedFunction(physics_pad_3d),
        Lux.Conv((5,5,5), 7=>32, pad=1), Lux.leakyrelu,
        Lux.WrappedFunction(physics_pad_3d),
        Lux.Conv((5,5,5), 32=>32, pad=1), Lux.leakyrelu,
        Lux.WrappedFunction(physics_pad_3d),
        Lux.Conv((3,3,3), 32=>16), Lux.leakyrelu,
        Lux.WrappedFunction(physics_pad_3d),
        Lux.Conv((3,3,3), 16=>1)
    )
end
