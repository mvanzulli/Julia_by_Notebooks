###########################
# Profile Copy DtH operation
############################
# load libraries 
using CUDA 
import BenchmarkTools as BT

# size of the matrix 
N = 2^14
mat = rand(Float32, N, N)


#=
# status and version info 
println("Cuda version info:\n")
CUDA.versioninfo()
println("MemStatus before allocation:\n")
CUDA.memory_status() 


# using @Elapsed macro
tDtH_elapsed = CUDA.@elapsed CUDA.@sync begin
    d_mat = cu(mat)
    # free device mem
    CUDA.unsafe_free!(d_mat)
end
println("\n")
println("\n tDtH_elapsed using cu() is $tDtH_elapsed \n")
# using copyto function 
tDtH_elapsed = CUDA.@elapsed CUDA.@sync begin
    d_mat = CuArray{Float32}(undef, (N,N))
    copyto!(d_mat, mat)
    # free device mem
    CUDA.unsafe_free!(d_mat)
end
println("\n tDtH_elapsed using copyto!() is $tDtH_elapsed \n")
=#

# using nsys profiler  
NVTX.@range "tDtH" CUDA.@profile begin
    d_mat = cu(mat)
    # free device mem
    CUDA.unsafe_free!(d_mat)
end


# # using benchmark tools macro
# BT.@benchmark CUDA.@sync begin
#     d_mat = cu(mat)
#     # free device mem
#     CUDA.unsafe_free!(d_mat)
# end


# println("MemStatus after allocation:\n")
# CUDA.memory_status() 
