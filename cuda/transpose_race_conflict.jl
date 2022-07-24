using CUDA
# Define tile dimension
const tile_dim = 16
"""
gpu_transpose(input::CuMatrix)

    Retrun the parllel transpose of a matrix

#Fields 
- input -- CuMatrix
"""
function gpu_transpose(input::CuMatrix)
    function kernel(input::AbstractMatrix{T}, output::AbstractMatrix{T}) where {T}
        # shared memory buffer so that operations to global memory are linear and can be coalesced
        block = @cuStaticSharedMem(T, (tile_dim, tile_dim))

        # read
        x = tile_dim * (blockIdx().x - 1) + threadIdx().x
        y = tile_dim * (blockIdx().y - 1) + threadIdx().y
        if x <= size(input, 1) && y <= size(input, 2)
            block[threadIdx().y, threadIdx().x] = input[x, y]
        end

        # write
        x = tile_dim * (blockIdx().y - 1) + threadIdx().x
        y = tile_dim * (blockIdx().x - 1) + threadIdx().y
        if x <= size(output, 1) && y <= size(output, 2)
            output[x, y] = block[threadIdx().x, threadIdx().y]
        end
        return
    end
    
    output = similar(input, reverse(size(input)))
    @cuda threads=(tile_dim, tile_dim) blocks=(cld(size(input, 1), tile_dim), cld(size(input, 2), tile_dim)) kernel(input, output)
    output
end

a = CuArray(reshape(1:1024, 32, 32))
b = gpu_transpose(a)
display(b)
println("Valid transpose: ", Array(b) == Array(a)')

# Extract the location of the sanatizer
CUDA.compute_sanitizer()

# then we execute 

#= 

/home/mvanzulli/.julia/artifacts/913584335ab836f9781a0325178d0949c193f50b/bin/compute-sanitizer --launch-timeout=0 --report-api-errors=no --tool=racecheck julia /home/mvanzulli/Repositories/gitHub/Julia_by_Notebooks/cuda/transpose_race_conflict.jl

=#
