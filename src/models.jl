#MODELS
function physics_pad_3d(x)
    x_h = cat(x[end:end, :, :, :, :], x, x[1:1, :, :, :, :]; dims=1)
    x_hw = cat(x_h[:, 1:1, :, :, :], x_h, x_h[:, end:end, :, :, :]; dims=2)
    x_hwl = cat(x_hw[:, :, 1:1, :, :], x_hw, x_hw[:, :, end:end, :, :]; dims=3)
    return x_hwl
end
struct StageAGCN{C1,C2,C3,D} <: Lux.AbstractLuxLayer
    conv1::C1; conv2::C2; conv3::C3; dense::D
end
function (m::StageAGCN)(x, ps, st)
    g = G_GLOBAL # Relies on global
    x1 = tanh.(m.conv1(g, x, ps.conv1, st.conv1)[1])
    x2 = tanh.(m.conv2(g, xf, ps.conv2, st.conv2)[1])
    x3 = tanh.(m.conv3(g, x2, ps.conv3, st.conv3)[1])
    y, st4 = m.dense(x3, ps.dense, st.dense)
    return y, (conv1=st.conv1, conv2=st.conv2, conv3=st.conv3, dense=st4)
end
Lux.initialparameters(rng::AbstractRNG, l::StageAGCN) = (conv1=Lux.initialparameters(rng,l.conv1), conv2=Lux.initialparameters(rng,l.conv2), conv3=Lux.initialparameters(rng,l.conv3), dense=Lux.initialparameters(rng,l.dense))
Lux.initialstates(rng::AbstractRNG, l::StageAGCN) = (conv1=Lux.initialstates(rng,l.conv1), conv2=Lux.initialstates(rng,l.conv2), conv3=Lux.initialstates(rng,l.conv3), dense=Lux.initialstates(rng,l.dense))
build_stageA_model(in_feats, rng) = StageAGCN(GCNConv(in_feats=>64), GCNConv(64=>64), GCNConv(64=>64), Lux.Dense(64=>1))
function build_stageB_model()
    Lux.Chain(
        Lux.WrappedFunction(physics_pad_3d), Lux.Conv((3,3,3), 7=>8, pad=0), Lux.relu,
        Lux.WrappedFunction(physics_pad_3d), Lux.Conv((3,3,3), 8=>8, pad=0), Lux.relu,
        Lux.WrappedFunction(physics_pad_3d), Lux.Conv((3,3,3), 8=>4, pad=0), Lux.relu,
        Lux.WrappedFunction(physics_pad_3d), Lux.Conv((3,3,3), 4=>1, pad=0)
    )
end
