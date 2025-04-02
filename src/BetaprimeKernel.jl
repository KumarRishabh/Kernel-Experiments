using KernelFunctions
using LinearAlgebra
using Revise
using SpecialFunctions  # Add this import for gamma function

include("mmd.jl")  # Assuming mmd.jl contains the MMD function

struct BetaPrimeKernel{T<:Real} <: KernelFunctions.Kernel
    α::T
    gamma_cache::Dict{Int,Float64}  # Cache for storing precomputed gamma values

    function BetaPrimeKernel(α::T=1.0) where {T<:Real}
        # Ensure α is valid (typically α > N-1 where N is matrix dimension)
        # This will be checked at evaluation time since we don't know N here
        new{T}(α)
        new{T}(α, Dict{Int,Float64}())
    end
end

# Helper function to compute special gamma function for SPD matrices
function cone_gamma(N::Int, alpha::Real)
    return exp(N*(N-1)/2 * log(2π) + sum(log.(gamma.(alpha .- collect(0:N-1) .+ 1))))
end

# Implementation of the kernel function
function (k::BetaPrimeKernel)(A::Matrix{<:Real}, B::Matrix{<:Real})
    # Check if matrices are square and same size
    n = size(A, 1)
    m = size(A, 2)
    
    if n != m || n != size(B, 1) || m != size(B, 2)
        throw(DimensionMismatch("Matrices must be square and of same size"))
    end
    
    # Set α = n to ensure positive-definiteness
    α = n
    
    # Check if α > n-1 (necessary condition)
    if α <= n-1
        throw(ArgumentError("α must be greater than n-1 where n is the matrix dimension"))
    end
    
    # Compute the normalization constant
    if !haskey(k.gamma_cache, n)
        k.gamma_cache[n] = cone_gamma(n, 2*α)
    end
    γ = k.gamma_cache[n]
    
    # Compute determinants
    detA = det(A)
    detB = det(B)
    detAB = det(A + B)
    
    # Handle potential numerical issues
    if detA <= 0 || detB <= 0 || detAB <= 0
        return 0.0  # Return zero for non-PD matrices
    end
    
    # Compute kernel value
    dets = (detA * detB / (detAB^2))^α
    return γ * dets
end

function (k::BetaPrimeKernel)(A::Matrix{<:Real}, B::Adjoint{<:Real, <:Matrix})
    return k(A, Matrix(B))  # Convert Adjoint to Matrix and use existing method
end

# Add the reverse case too
function (k::BetaPrimeKernel)(A::Adjoint{<:Real, <:Matrix}, B::Matrix{<:Real})
    return k(Matrix(A), B)  # Convert Adjoint to Matrix and use existing method
end

# And for completeness, the case when both are Adjoint
function (k::BetaPrimeKernel)(A::Adjoint{<:Real, <:Matrix}, B::Adjoint{<:Real, <:Matrix})
    return k(Matrix(A), Matrix(B))  # Convert both Adjoints to Matrices
end

# Extension for vector inputs (if needed)
function (k::BetaPrimeKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    error("BetaPrimeKernel is designed for SPD matrices, not vectors")
end


# Example usage:
# Create the kernel
bp_kernel = BetaPrimeKernel(5.0)  # α = 5.0

# Generate some SPD matrices for testing
function generate_spd_matrix(n)
    A = randn(n, n)
    return A * A' + I  # Ensures positive-definiteness
end

# Generate some test matrices
n = 3  # Matrix dimension
matrices_X = [generate_spd_matrix(n) for _ in 1:10]
matrices_Y = [generate_spd_matrix(n) for _ in 1:10]

function MMD(X, Y; sigma=1.0, kernel=GaussianKernel())
    # X and Y are the two distributions
    # sigma is the bandwidth of the Gaussian kernel
    # The function returns the MMD value
    # Compute the Gram matrix for X using the specified kernel
    K_X = kernel.(X, X)
    
    # Compute the Gram matrix for Y using the specified kernel
    K_Y = kernel.(Y, Y)
    
    # Compute the cross Gram matrix between X and Y using the specified kernel
    K_XY = kernel.(X, Y)
    
    # Compute the MMD value
    mmd_value = mean(K_X) + mean(K_Y) - 2 * mean(K_XY)
    return mmd_value
end


# Use with your MMD function
mmd_value = MMD(matrices_X, matrices_Y; kernel=BetaPrimeKernel())