using Images
#using Distributions
function get_target_sizes(array::AbstractArray)
    s = size(array)
    @assert length(s) == 4
    h, w, c, n = s
    return h, w, c, n
end


function load_image(path::String, size::Tuple{Int64,Int64}=(256, 256))
    im = load(path)
    im = channelview(im)
    im = reshape(im, size..., 3, 1)
    return float.(im)
end

function read_batch(paths::Vector, n::Int=10)
    X = Array{Float32}(undef, 256, 256, 3, n)
    max = length(paths)
    samples = rand(1:max, n)
    for i in 1:n
        im = load_image(paths[samples[i]])
        X[:, :, :, i] = im[:, :, :, 1]
    end
    return X
end

function Display(X::Array, i::Int)
    im = reshape(X[:, :, :, i], 3, 256, 256)
    im = colorview(RGB, im)
    display(im)
    return im
end

