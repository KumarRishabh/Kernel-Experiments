using Random 
using LinearAlgebra
using Statistics
using KernelFunctions
using Distributions
using Plots

function generate_GMM(n_samples, n_clusters, n_dimensions, means, cov)
    X = zeros(n_samples, n_dimensions)  # Pre-allocate the data matrix
    # Generate data from the mixture of gaussians
    for i in 1:n_samples
        # Randomly select a cluster
        cluster = rand(1:n_clusters)
        # Generate a sample from the selected cluster
        X[i, :] = rand(MvNormal(means[cluster], cov[cluster]))
        # Store the sample in the data matrix
    end
    return X
end

# Generate 1000 samples from the mixture of gaussians

# Set random seed for reproducibility
Random.seed!(1234)

# Number of samples
n_samples = 1000
# Number of clusters
n_clusters = 9
# Number of dimensions
n_dimensions = 2

# set the mean of the clusters
means = [Float64[i*5, j*5] for i in 1:3 for j in 1:3]
# set the covariance of the clusters
cov = [0.3I for _ in 1:n_clusters]
X = generate_GMM(n_samples, n_clusters, n_dimensions, means, cov)

# plot the data 
scatter(X[:, 1], X[:, 2], title="Mixture of Gaussians", xlabel="\$X_1\$", ylabel="\$X_2\$", legend=false, markersize=2)
# Define the Gaussian (RBF) kernel with lengthscale parameter ℓ

# rand(MvNormal([1, 1, 1], 0.4I))

staggered_cov = [0.3I for _ in 1:n_clusters]
# Make the covariance matrices staggered
for i in 1:n_clusters
    staggered_cov[i] = 0.3I + 0.1 * (-1) * I
end

# Generate data from the staggered mixture of gaussians
X_staggered = generate_GMM(n_samples, n_clusters, n_dimensions, means, staggered_cov)

scatter(X_staggered[:, 1], X_staggered[:, 2], title="Staggered Mixture of Gaussians", xlabel="\$X_1\$", ylabel="\$X_2\$", legend=false, markersize=2)