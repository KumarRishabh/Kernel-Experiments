# PosDefManifolds.jl was build by Congedo et. al. with the help of Saloni Jain, from IITKGP, India (my batchmate)
# TODO: Add kernel functions functionality to Manifolds.jl

# First install the required packages
using PosDefManifold
using LinearAlgebra
using Distributions
using Random
using Plots

# Define the dimension of SPD matrices
n = 3  # For 3x3 matrices

# Sample from the manifold
# Method 1: Using Wishart distribution (most common)
k = 2*n  # Degrees of freedom (>= n)
Σ = Matrix{Float64}(I, n, n)  # Scale matrix (typically identity)
nsamples = 100  # Number of samples


# Generate samples
samples = randPosDefMat(Hermitian, n, k, nsamples)  # For real symmetric PD matrices
R=randP(10, df=10, eigvalsSNR=1000) 
# Display a few samples
for i in 1:min(3, nsamples)
    println("Sample $i:")
    display(samples[i])
    println("Eigenvalues: ", eigvals(samples[i]))
    println()
end

# Method 2: Uniform sampling in the tangent space and projecting back
# This is available in more recent versions
base_point = Matrix{Float64}(I, n, n)  # Identity as base point
samples_uniform = randP(n, nsamples; tangentSpace=true, meanISR=base_point)

# Method 3: Specify your own distribution of eigenvalues
# For a specific distribution of eigenvalues
eigdist = [rand() * 5 + 1 for _ in 1:n]  # Random eigenvalues between 1 and 6
samples_eigdist = randP(eigdist, nsamples)

# Visualize the samples (for 2x2 matrices)
if n == 2
    # Extract eigenvalues
    λ1 = [eigvals(samples[i])[1] for i in 1:nsamples]
    λ2 = [eigvals(samples[i])[2] for i in 1:nsamples]
    
    scatter(λ1, λ2, 
        title="Distribution of eigenvalues", 
        xlabel="λ₁", ylabel="λ₂", 
        legend=false)
end

# For higher dimensions, visualize using the log-determinant and trace
logdet_values = [logdet(samples[i]) for i in 1:nsamples]
trace_values = [tr(samples[i]) for i in 1:nsamples]

scatter(logdet_values, trace_values,
    title="Distribution of samples",
    xlabel="log(det)", ylabel="trace",
    legend=false)


# Sample from the manifold using a gaussian distribution
# Define the mean and covariance for the Gaussian distribution
mean = Matrix{Float64}(I, n, n)  # Identity matrix as mean
cov = 0.1 * Matrix{Float64}(I, n, n)  # Small covariance

# Generate samples from the Gaussian distribution
samples_gaussian = randn(n, n, nsamples)  # Random samples
