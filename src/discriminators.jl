using MLDatasets: MNIST
using Flux
using CUDA
using Zygote
using UnicodePlots
using Logging
import Statistics
include("utils.jl")


# DISCRIMINATOR 1 ---
"""
"""
d1 = Chain(
    # encoder
    Conv((4, 4), 3 => 32 * 2, relu, stride=2),
    Conv((4, 4), 32 * 2 => 32 * 4, relu, stride=2),
    Conv((4, 4), 32 * 4 => 32 * 8, relu, stride=2),
    Conv((4, 4), 32 * 8 => 32 * 16, relu, stride=2),
    Conv((4, 4), 32 * 16 => 1, relu, stride=2),
    x -> σ(x)
) |> gpu

# DISCRIMINATOR 2 ------ 
"""
"""
d2 = Chain(
    # encoder
    Conv((4, 4), 3 => 32 * 2, relu, stride=2),
    Conv((4, 4), 32 * 2 => 32 * 4, relu, stride=2),
    Conv((4, 4), 32 * 4 => 32 * 8, relu, stride=2),
    Conv((4, 4), 32 * 8 => 32 * 16, relu, stride=2),
    Conv((4, 4), 32 * 16 => 1, relu, stride=2),
    x -> σ(x)
) |> gpu



# TRAINERS 

function train_discriminator!(discriminator, opt, real_image, fake_image)

    # FAKE DATA ----
    # should predict 0
    ps = Flux.params(discriminator)
    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(fake_image)
        h, w, c, n = get_target_sizes(preds)
        fake_targets = zeros(eltype(fake_image), h, w, c, n) |> gpu
        loss_fake = Flux.Losses.binarycrossentropy(preds, fake_targets)

        preds = discriminator(real_image)
        h, w, c, n = get_target_sizes(preds)
        fake_targets = ones(eltype(real_image), h, w, c, n) |> gpu
        loss_real = Flux.Losses.binarycrossentropy(preds, fake_targets)
        return loss_fake + loss_real
    end
    # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
    grads = pullback(one(loss))

    # Update the parameters of the discriminator with the gradients we calculated above
    Flux.update!(opt, ps, grads)

    #return loss_fake, loss_real, loss
    return loss
end

