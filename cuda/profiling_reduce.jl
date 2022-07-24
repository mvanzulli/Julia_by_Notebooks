using CUDA, Test
"""
reduce_grid_atomic(op, a, b)

    reduce kernel using atomic operataions 

## Fields
op:: reduction operation function 
a:: CuArray to reduce
b:: variable where reduce result is stored 
"""
function reduce_grid_atomic(op::Function,
                            a::AbstractArray{T},
                            b::AbstractArray{T}) where T
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
"""
reduce_grid_atomic(op, a, b)

    reduce kernel using shared memory

## Fields
op:: reduction operation function 
a:: CuArray to reduce
b:: variable where reduce result is stored 
"""

function reduce_grid_shared(op::Function,
                            a::AbstractArray{T},
                            b::AbstractArray{T}) where {T}
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


"""
my_reduce(op, a) 

my_reduce functions to launch the kernel 

## Fields
op:: reduction operation function 
a:: CuArray to reduce
"""
function my_reduce(op::Function, a::AbstractArray{T}) where {T}
    # launch atomic reduction
    a_atomic = copy(a) 
    b_atomic = CUDA.zeros(T, 1)

    kernel_atomic = @cuda(
        launch=false,
        reduce_grid_atomic(+, a_atomic, b_atomic)
    ) 

    config = launch_configuration(kernel_atomic.fun)
    threads_config = min(config.threads, length(a))
    threads = 1024
    blocks = cld(length(a_atomic), threads*2)
    # println("threads_config atomic is $threads_config")
    # println("threads atomic is $threads")
    # println("blocks atomic is $blocks")
    @cuda(
        threads=threads,
        blocks=blocks,
        reduce_grid_atomic(op, a_atomic, b_atomic)
    ) 
    # launch shared memory  reduction
    b_shared = CUDA.zeros(T, 1)

    kernel_shared = @cuda(
        launch=false,
        reduce_grid_shared(+, a, b_shared)
    ) 

    config = launch_configuration(kernel_shared.fun)
    threads_config = min(config.threads, length(a))
    threads = 1024
    blocks = cld(length(a), threads*2)
    # println("threads_config shared is $threads_config")
    # println("threads shared is $threads")
    # println("blocks shared is $blocks")
    @cuda(
        threads=threads,
        blocks=blocks,
        reduce_grid_shared(op, a, b_shared)
    ) 
    # test outputs
    @assert b_shared ≈ b_atomic

    CUDA.@allowscalar b_atomic[]
end


"""
main()

function to launch the kernel
"""
function main()
    N = 1024
    c_a = rand(N, N, 10)
    d_a = CuArray(c_a)
    @test my_reduce(+, d_a) ≈ sum(c_a)

    # profile it 
    CUDA.@profile begin 
        NVTX.@range "my_reduce" my_reduce(+, d_a)
        NVTX.@range "my_reduce" my_reduce(+, d_a)
    end
end
# execute the function
# main()  
