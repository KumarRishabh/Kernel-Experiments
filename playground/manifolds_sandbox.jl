using Manifolds
using LinearAlgebra
using Statistics

using KernelFunctions
# TODO: Add kernel functions functionality to Manifolds.jl

# Instantiate the SPD manifold
M = SymmetricPositiveDefinite(2)
# Create a random point on the manifold
n_points = 100
X = rand(M, n_points)

base = Matrix{Float64}(I, 3, 3)

# Function to sample a symmetric matrix in the tangent space at the base.
# This corresponds to sampling from a Euclidean Gaussian in the space of symmetric matrices.
function sample_tangent(base, σ)
    n = size(base, 1)
    # Generate a random matrix and symmetrize it.
    A = randn(n, n)
    A = 0.5 * (A + A')
    return σ * A
end

# Function to sample one SPD matrix from a "Gaussian" on the manifold.
function sample_gaussian_spd(M, base, σ)
    v = sample_tangent(base, σ)     # sample a tangent vector according to Euclidean Gaussian
    return exp(M, base, v)          # map the tangent vector to the manifold via the exponential map
end

# Number of samples you want
nsamples = 100

# Standard deviation in the tangent space (controls spread on the manifold)
σ = 1.0

# Generate a collection of samples on the SPD manifold
samples_gaussian = [sample_gaussian_spd(M, base, σ) for i in 1:nsamples]

# For example, visualize the distribution of the log-determinant and trace of the samples:
logdet_values = [logdet(s) for s in samples_gaussian]
trace_values = [tr(s) for s in samples_gaussian]

scatter(logdet_values, trace_values,
    title="Gaussian SPD Samples",
    xlabel="log(det)", ylabel="trace",
    legend=false, markersize=4)

# 

# For 2×2 SPD matrices, we can visualize them using their eigenvalues 
# or using the Log-Euclidean metric embedding


# ----- PLOTTING OPTIONS -----
# # Option 1: Plot using eigenvalues
# eigenvalues = zeros(n_points, 2)
# for i in 1:n_points
#     eigenvalues[i, :] = eigvals(X[i])
# end

# scatter(eigenvalues[:, 1], eigenvalues[:, 2], 
#     title="SPD Matrices - Eigenvalue Representation",
#     xlabel="λ₁", ylabel="λ₂",
#     legend=false, markersize=4)

# # Option 2: Use PCA on the Log-Euclidean metric embedding
# function log_euclidean_embedding(X)
#     n = length(X)
#     base_point = mean(X) # Use arithmetic mean as reference point
#     embedded = zeros(n, 3) # For 2×2 matrices, the tangent space is 3-dimensional
    
#     for i in 1:n
#         L = log(X[i]) # Matrix logarithm
#         # Store the unique elements (for 2×2, there are 3: L[1,1], L[1,2], L[2,2])
#         embedded[i, 1] = L[1, 1]
#         embedded[i, 2] = L[1, 2]
#         embedded[i, 3] = L[2, 2]
#     end
    
#     return embedded
# end

# embedded = log_euclidean_embedding(X)

# # Apply PCA to reduce to 2D for visualization (simplified here)
# using MultivariateStats
# pca_model = fit(PCA, embedded', maxoutdim=2)
# projected = transform(pca_model, embedded')

# scatter(projected[1, :], projected[2, :], 
#     title="SPD Matrices - Log-Euclidean Embedding with PCA",
#     xlabel="PC1", ylabel="PC2",
#     legend=false, markersize=4)

# # Option 3: Use t-SNE for visualization
# using TSne
# perplexity = min(30.0, n_points/3 - 1)
# embedded_tsne_3d = try
#     # Try with modified parameters
#     tsne(embedded, 3, 50, 1000, perplexity)
# catch e
#     println("t-SNE in 3D failed, using PCA instead: ", e)
#     # Fallback to PCA if t-SNE fails
#     pca_model_3d = fit(PCA, embedded', maxoutdim=3)
#     transform(pca_model_3d, embedded')'
# end

# if size(embedded_tsne_3d, 2) == 3
#     # Plot in 3D if we have 3 dimensions
#     scatter3d(embedded_tsne_3d[:, 1], embedded_tsne_3d[:, 2], embedded_tsne_3d[:, 3],
#         title="SPD Matrices - 3D Visualization",
#         xlabel="Dim 1", ylabel="Dim 2", zlabel="Dim 3",
#         legend=false, markersize=4)
# else
#     println("Could not create 3D visualization, dimensions: ", size(embedded_tsne_3d))
# end

# # Alternative: Use the log-Euclidean embedding directly (it's already 3D for 2×2 SPD matrices)
# scatter3d(embedded[:, 1], embedded[:, 2], embedded[:, 3],
#     title="SPD Matrices - Log-Euclidean Space",
#     xlabel="log(X)[1,1]", ylabel="log(X)[1,2]", zlabel="log(X)[2,2]",
#     legend=false, markersize=4)

