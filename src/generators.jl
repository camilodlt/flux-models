using MLDatasets: MNIST
using Flux
using CUDA
using Zygote
using UnicodePlots
using Logging
import StatsBase
include("utils.jl")


# GENERATOR 1 ---
"""
"""

# G₁ = Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
#     Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
#     Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
#     Dense(1024, n_features, tanh)) |> gpu


g1 = Chain(
    # encoder
    Conv((7, 7), 3 => 32, relu, stride=1, pad=1),
    InstanceNorm(32),
    Conv((3, 3), 32 => 32 * 2, relu, stride=2, pad=1),
    InstanceNorm(32 * 2),
    Conv((3, 3), 32 * 2 => 32 * 4, relu, stride=2, pad=1),
    InstanceNorm(32 * 4),
    Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=2, pad=1),
    InstanceNorm(32 * 4),

    # embedding
    SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1, pad=1), (mx, x) -> mx .+ x), # L5
    InstanceNorm(32 * 4),
    SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1, pad=1), (mx, x) -> mx .+ x),
    InstanceNorm(32 * 4),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 2, relu, stride=1), (mx, x) -> mx .+ x),

    # decoder
    ConvTranspose((4, 4), 32 * 4 => 32 * 2, relu, stride=2, pad=1),
    InstanceNorm(32 * 2),
    ConvTranspose((4, 4), 32 * 2 => 3, relu, stride=4),
    InstanceNorm(3)
    #ConvTranspose((3,3), 32*2 => 3, relu, stride = 1, pad =1)

) |> gpu

# GENERATOR 2 ------ 
"""
"""

g2 = Chain(
    # encoder
    Conv((7, 7), 3 => 32, relu, stride=1, pad=1),
    InstanceNorm(32),
    Conv((3, 3), 32 => 32 * 2, relu, stride=2, pad=1),
    InstanceNorm(32 * 2),
    Conv((3, 3), 32 * 2 => 32 * 4, relu, stride=2, pad=1),
    InstanceNorm(32 * 4),
    Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=2, pad=1),
    InstanceNorm(32 * 4),

    # embedding
    SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1, pad=1), (mx, x) -> mx .+ x), # L5
    InstanceNorm(32 * 4),
    SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1, pad=1), (mx, x) -> mx .+ x),
    InstanceNorm(32 * 4),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 4, relu, stride=1), (mx, x) -> mx .+ x),
    #SkipConnection(Conv((3, 3), 32 * 4 => 32 * 2, relu, stride=1), (mx, x) -> mx .+ x),

    # decoder
    ConvTranspose((4, 4), 32 * 4 => 32 * 2, relu, stride=2, pad=1),
    InstanceNorm(32 * 2),
    ConvTranspose((4, 4), 32 * 2 => 3, tanh, stride=4),
    InstanceNorm(3)
    #ConvTranspose((3,3), 32*2 => 3, relu, stride = 1, pad =1)

) |> gpu

# TRAINERS 
function train_generator!(generator, discriminator, opt, real_image)

    # FAKE DATA ----
    # should not be detected by the discriminator
    ps = Flux.params(generator)
    loss, pullback = Zygote.pullback(ps) do
        global fake_image = generator(real_image)
        preds = discriminator(fake_image)
        loss = Flux.Losses.binarycrossentropy(preds, 1.0)
    end

    # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
    grads = pullback(one(loss))


    # Update the parameters of the discriminator with the gradients we calculated above
    Flux.update!(opt, ps, grads)

    return loss, grads, ps
end


# 
# g1 : horse => zebra
# d1 : horse
# 
# g2 : zebra => horse
# d2 : zebra






