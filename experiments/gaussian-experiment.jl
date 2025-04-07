# Create an experiment to test the BetaPrimeKernel for the two sample testing problem with Gaussian Covariance Matrices

using Random
using LinearAlgebra
using Statistics
using KernelFunctions
using Distributions
using Test
include("../src/mmd.jl")

# Set random seed for reproducibility

Random.seed!(1234)

# Sample X: 100 observations from a standard normal distribution
X = randn(2, 100)

# Sample Y: 100 observations from a normal distribution with mean 1
Y = randn(2, 100) .+ 1

# Define the Gaussian (RBF) kernel with lengthscale parameter ℓ
ℓ = 1.0
k = with_lengthscale(SqExponentialKernel(), ℓ)
# Compute kernel matrices
