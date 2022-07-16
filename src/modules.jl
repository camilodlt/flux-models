using MLDatasets: MNIST
using Flux
using CUDA
using Zygote
using UnicodePlots
using Logging
using Images
using Plots
include("discriminators.jl")
include("generators.jl")
include("utils.jl")
@info "Package Started"

# PLOT IMAGE ------
path₁ = "datasets/monet2photo/trainA/"
images₁ = readdir(path₁)
images₁ = joinpath.(path₁, images₁)
#im₁ = load_image(path₁) |> gpu

path₂ = "datasets/monet2photo/trainB/"
images₂ = readdir(path₂)
images₂ = joinpath.(path₂, images₂)
#im₂ = load_image(path₂) |> gpu

# LOAD BATCHES ------
b₁ = read_batch(images₁) |> gpu
b₂ = read_batch(images₂) |> gpu

# CONFIG ------ 
lr_g = 2e-1          # Learning rate of the generator network
lr_d = 1 # 2e-1          # Learning rate of the discriminator network
batch_size = 1    # batch size
num_epochs = 30  # Number of epochs to train for
output_period = 1 # Period length for plots of generator samples
n_features = 256 * 256 # Number of pixels in each sample of the MNIST dataset
latent_dim = 100    # Dimension of latent space
opt_dscr₁ = ADAM(lr_d)# Optimizer for the discriminator 1 
opt_dscr₂ = ADAM(lr_d)# Optimizer for the discriminator 1 

opt_gen₁ = ADAM(lr_g) # Optimizer for the generator
opt_gen₂ = ADAM(lr_g) # Optimizer for the generator
do_cycle_consistency = false

# lossvec_gen = zeros(num_epochs)
# lossvec_dscr = zeros(num_epochs)

@info "Training"
for n in 1:num_epochs
    loss_sum_gen₁, loss_sum_gen₂, loss_sum_dscr₁, loss_sum_dscr₂ = (0.0f0 for i in 1:4)
    b₁ = read_batch(images₁) |> gpu
    b₂ = read_batch(images₂) |> gpu
    for (real_image₁, real_image₂) in zip([b₁], [b₂])
        #* GENERATORS ---
        # Generate 1 => 2
        l, g, p = train_generator!(g2, d2, opt_gen₂, real_image₁)
        @info "Loss Generator 1 => 2  : $l"

        # Generate 2 => 1
        l, g, p = train_generator!(g1, d1, opt_gen₁, real_image₂)
        @info "Loss Generator 2 => 1  : $l"

        #* DISCRIMINATORS --- 
        # Discriminate 1
        fake_image₁ = g1(real_image₂)
        l = train_discriminator!(d1, opt_dscr₁, real_image₁, fake_image₁)
        @info "Loss Discriminator 2 : $l"

        # Discriminate 2    
        fake_image₂ = g2(real_image₁)
        l = train_discriminator!(d2, opt_dscr₂, real_image₂, fake_image₂)
        @info "Loss Discriminator 2 : $l"


        Display(fake_image₂, 2)

        sleep(15)
        #* CYCLE CONSISTENCY
        if do_cycle_consistency
            # fake_image₁ => real_image₂
            train_generator!(g2, d2, opt_gen₂, fake_image₁)
            recreated_image₂ = g2(fake_image₁)
            train_discriminator!(d2, opt_dscr₂, real_image₂, recreated_image₂)

            # fake_image₂ => real_image₁
            train_generator!(g1, d1, opt_gen₁, fake_image₂)
            recreated_image₁ = g1(fake_image₂)
            train_discriminator!(d1, opt_dscr₁, real_image₁, recreated_image₁)


            Display(recreated_image₁, 1)
        end
    end

    # # Add the per-sample loss of the generator and discriminator
    # lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
    # lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]

    # if n % output_period == 0
    #     @show n
    #     noise = randn(latent_dim, 4) |> gpu
    #     fake_data = reshape(generator(noise), 28, 4 * 28)
    #     p = heatmap(fake_data, colormap=:inferno)
    #     print(p)
    # end
end
