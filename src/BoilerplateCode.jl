using KernelFunctions
using Random
using LinearAlgebra

# TODO: Test the boilerplate code and gain understanding of the MMD and permutation test
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
K_XX = kernelmatrix(k, ColVecs(X))
K_YY = kernelmatrix(k, ColVecs(Y))
K_XY = kernelmatrix(k, ColVecs(X), ColVecs(Y))

function compute_mmd(K_XX, K_YY, K_XY)
    n = size(K_XX, 1)
    m = size(K_YY, 1)
    mmd_sq = (sum(K_XX) - tr(K_XX)) / (n * (n - 1)) +
             (sum(K_YY) - tr(K_YY)) / (m * (m - 1)) -
             2 * sum(K_XY) / (n * m)
    return sqrt(mmd_sq)
end

# Calculate the MMD statistic
mmd_value = compute_mmd(K_XX, K_YY, K_XY)
println("MMD Statistic: ", mmd_value)

function permutation_test(X, Y, k, num_permutations=1000)
    n = size(X, 2)
    m = size(Y, 2)
    Z = hcat(X, Y)
    mmd_orig = compute_mmd(
        kernelmatrix(k, ColVecs(X)),
        kernelmatrix(k, ColVecs(Y)),
        kernelmatrix(k, ColVecs(X), ColVecs(Y))
    )
    count = 0
    for i in 1:num_permutations
        perm = randperm(n + m)
        X_perm = Z[:, perm[1:n]]
        Y_perm = Z[:, perm[n+1:end]]
        mmd_perm = compute_mmd(
            kernelmatrix(k, ColVecs(X_perm)),
            kernelmatrix(k, ColVecs(Y_perm)),
            kernelmatrix(k, ColVecs(X_perm), ColVecs(Y_perm))
        )
        if mmd_perm >= mmd_orig
            count += 1
        end
    end
    p_value = count / num_permutations
    return p_value
end

# Perform permutation test
p_value = permutation_test(X, Y, k)
println("p-value: ", p_value)
