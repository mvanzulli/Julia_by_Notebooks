using BenchmarkTools, CUDA, Test

#######################
# REDUCE GRID ATOMIC
#######################
function reduce_grid_atomic(op, a, b)
    num_elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    
    #parallel reduction of values in a block (stride or distance between each thread reduction) 
    stride_threads = 1
    # parallel reduction between blocks has a stride of 
    stride_blocks = (block - 1)*num_elements

    
    # while still have elements to reduce 
    while stride_threads < num_elements
        # add a barrier to sync threads
        sync_threads()
        # compute index to reduce 
        index = 2*stride_threads*(thread - 1) + 1 
        # check index and index + d are inbounds a
        @inbounds if index ≤ num_elements && index + stride_threads + stride_blocks ≤ length(a)
#             CUDA.@cuprintln ("thread $thread: a[$index] + a[$(index + stride_blocks)] = $(a[index] + a[index + stride_blocks])")
            a[stride_blocks + index] = op(a[index + stride_blocks], a[index + stride_threads + stride_blocks])
        end
        stride_threads *= 2
    end
    # do attomic operatios with the first entry of ech block (sum through each block)
    if thread == 1 
        CUDA.@atomic b[] = op(b[], a[stride_blocks + 1])
    end
    return nothing
end

# define test inputs
size = 2048
c_a = 1:size
d_a = CuArray(1:size)
d_b = CuArray([0])
# lunch kernel
@cuda(
    threads = 256,
    blocks = cld(256, size),
    reduce_grid_atomic(+, d_a, d_b)
    )
# test the result 
using Test
CUDA.@allowscalar d_b
@test CUDA.@allowscalar d_b[] == sum(c_a)


#######################
# REDUCE SHARED MEMORY
#######################
function reduce_grid_shared(op, a::AbstractArray{T}, b) where {T}
    num_elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    #parallel reduction of values in a block (stride or distance between each thread reduction) 
    stride_threads = 1
    # parallel reduction between blocks has a stride of 
    stride_blocks = (block - 1)*num_elements
    
    # shared mem to buffer the a elements
    shared = @cuStaticSharedMem(T, (2048,))
    @inbounds shared[thread] = a[thread + stride_blocks]
    @inbounds shared[thread + blockDim().x] = a[thread + stride_blocks + blockDim().x]
 
    # while still have elements to reduce 
    while stride_threads < num_elements
        # add a barrier to sync threads
        sync_threads()
        # compute index to reduce 
        index = 2*stride_threads*(thread - 1) + 1 
        # check index and index + d are inbounds a
        @inbounds if index ≤ num_elements && index + stride_threads + stride_blocks ≤ length(a)
            shared[index] = op(shared[index], shared[index + stride_threads])
        end
        stride_threads *= 2
    end
    # do attomic operatios with the first entry of ech block reduction at shared 
    if thread == 1 
        CUDA.@atomic b[] = op(b[], shared[1])
    end
    return nothing
end

##########################
# TEST ALL IMPLEMENTATIONS
#########################

# define test inputs
c_a = 1:16
d_a = CuArray(1:16)
d_b = CuArray([0])
# lunch kernel shared
@cuda(
    threads = 4,
    blocks = 2,
    reduce_grid_atomic(+, d_a, d_b)
    )
# test the result 
@test CUDA.@allowscalar d_b[] == sum(c_a)
# re-define test inputs
CUDA.unsafe_free!(d_b)
CUDA.unsafe_free!(d_a)
d_a = CuArray(1:16)
d_b = CuArray([0])

# lunch kernel shared
@cuda(
    threads = 4,
    blocks = 2,
    reduce_grid_shared(+, d_a, d_b)
    )
@test CUDA.@allowscalar d_b[] == sum(c_a)
#=
##########################
# Banchmark both
#########################
N = 2048
    @show @benchmark CUDA.@sync @cuda( 
        threads = 1024, 
        blocks = 2,
        reduce_grid_shared(+,   $(CUDA.rand(N, N)), $(CUDA.rand(N, N)))
        )
@show @benchmark CUDA.@sync @cuda( 
    threads = 1024, 
    blocks = 2,
    reduce_grid_atomic(+,   $(CUDA.rand(N, N)), $(CUDA.rand(N, N)))
    )
=# using BenchmarkTools, CUDA, Test

#######################
# REDUCE GRID ATOMIC
#######################
function reduce_grid_atomic(op, a, b)
    num_elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    
    #parallel reduction of values in a block (stride or distance between each thread reduction) 
    stride_threads = 1
    # parallel reduction between blocks has a stride of 
    stride_blocks = (block - 1)*num_elements

    
    # while still have elements to reduce 
    while stride_threads < num_elements
        # add a barrier to sync threads
        sync_threads()
        # compute index to reduce 
        index = 2*stride_threads*(thread - 1) + 1 
        # check index and index + d are inbounds a
        @inbounds if index ≤ num_elements && index + stride_threads + stride_blocks ≤ length(a)
#             CUDA.@cuprintln ("thread $thread: a[$index] + a[$(index + stride_blocks)] = $(a[index] + a[index + stride_blocks])")
            a[stride_blocks + index] = op(a[index + stride_blocks], a[index + stride_threads + stride_blocks])
        end
        stride_threads *= 2
    end
    # do attomic operatios with the first entry of ech block (sum through each block)
    if thread == 1 
        CUDA.@atomic b[] = op(b[], a[stride_blocks + 1])
    end
    return nothing
end

# define test inputs
size = 2048
c_a = 1:size
d_a = CuArray(1:size)
d_b = CuArray([0])
# lunch kernel
@cuda(
    threads = 256,
    blocks = cld(),
    reduce_grid_atomic(+, d_a, d_b)
    )
# test the result 
using Test
CUDA.@allowscalar d_b
@test CUDA.@allowscalar d_b[] == sum(c_a)


#######################
# REDUCE SHARED MEMORY
#######################
function reduce_grid_shared(op, a::AbstractArray{T}, b) where {T}
    num_elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    #parallel reduction of values in a block (stride or distance between each thread reduction) 
    stride_threads = 1
    # parallel reduction between blocks has a stride of 
    stride_blocks = (block - 1)*num_elements
    
    # shared mem to buffer the a elements
    shared = @cuStaticSharedMem(T, (2048,))
    @inbounds shared[thread] = a[thread + stride_blocks]
    @inbounds shared[thread + blockDim().x] = a[thread + stride_blocks + blockDim().x]
 
    # while still have elements to reduce 
    while stride_threads < num_elements
        # add a barrier to sync threads
        sync_threads()
        # compute index to reduce 
        index = 2*stride_threads*(thread - 1) + 1 
        # check index and index + d are inbounds a
        @inbounds if index ≤ num_elements && index + stride_threads + stride_blocks ≤ length(a)
            shared[index] = op(shared[index], shared[index + stride_threads])
        end
        stride_threads *= 2
    end
    # do attomic operatios with the first entry of ech block reduction at shared 
    if thread == 1 
        CUDA.@atomic b[] = op(b[], shared[1])
    end
    return nothing
end

##########################
# TEST ALL IMPLEMENTATIONS
#########################

# define test inputs
c_a = 1:16
d_a = CuArray(1:16)
d_b = CuArray([0])
# lunch kernel shared
@cuda(
    threads = 4,
    blocks = 2,
    reduce_grid_atomic(+, d_a, d_b)
    )
# test the result 
@test CUDA.@allowscalar d_b[] == sum(c_a)
# re-define test inputs
CUDA.unsafe_free!(d_b)
CUDA.unsafe_free!(d_a)
d_a = CuArray(1:16)
d_b = CuArray([0])

# lunch kernel shared
@cuda(
    threads = 4,
    blocks = 2,
    reduce_grid_shared(+, d_a, d_b)
    )
@test CUDA.@allowscalar d_b[] == sum(c_a)
#=
##########################
# Banchmark both
#########################
N = 2048
    @show @benchmark CUDA.@sync @cuda( 
        threads = 1024, 
        blocks = 2,
        reduce_grid_shared(+,   $(CUDA.rand(N, N)), $(CUDA.rand(N, N)))
        )
@show @benchmark CUDA.@sync @cuda( 
    threads = 1024, 
    blocks = 2,
    reduce_grid_atomic(+,   $(CUDA.rand(N, N)), $(CUDA.rand(N, N)))
    )
=# 