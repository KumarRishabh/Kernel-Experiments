# implement the mmd function here 
module MMD
    using KernelFunctions
    # using the mmd function from the paper "Maximum Mean Discrepancy for 
    # testing independence of random variables" by Gretton et al. (2005)

    # The MMD function computes the maximum mean discrepancy (MMD) between two
    # distributions using the Gaussian kernel.
    # TODO: Implement the betaprime kernel as discussed in the arxiv by Mostajeran et. al.
    function betaprimekernel(x, y; alpha=1.0, beta=1.0)
        # x and y are the two distributions
        # alpha and beta are the parameters of the beta prime kernel
        # The function returns the beta prime kernel value
        return (x^alpha) * (y^beta) / (x + y)^(alpha + beta)
    end

    # The MMD is a measure of the difference between two probability distributions.
    # The MMD is calculated according to the following formula:
    # MMD(X, Y) = ||μ_X - μ_Y||^2
    # where μ_X and μ_Y are the mean embeddings of the distributions X and Y, respectively.
    # The mean embedding is computed using the kernel function K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    # The MMD is a measure of the distance between the two distributions in the reproducing kernel Hilbert space (RKHS) induced by the kernel function.

    function MMD(X, Y; sigma=1.0, kernel=GaussianKernel())
        # X and Y are the two distributions
        # sigma is the bandwidth of the Gaussian kernel
        # The function returns the MMD value
        # Compute the Gram matrix for X using the specified kernel
        K_X = kernel.(X, X')
        
        # Compute the Gram matrix for Y using the specified kernel
        K_Y = kernel.(Y, Y')
        
        # Compute the cross Gram matrix between X and Y using the specified kernel
        K_XY = kernel.(X, Y')
        
        # Compute the MMD value
        mmd_value = mean(K_X) + mean(K_Y) - 2 * mean(K_XY)
        return mmd_value
    end



    function mmd(X, Y; sigma=1.0)
        # X and Y are the two distributions
        # sigma is the bandwidth of the Gaussian kernel
        # The function returns the MMD value

        # Compute the Gram matrix for X
        K_X = exp.(-pairwise(Euclidean(), X, X) ./ (2 * sigma^2))
        
        # Compute the Gram matrix for Y
        K_Y = exp.(-pairwise(Euclidean(), Y, Y) ./ (2 * sigma^2))
        
        # Compute the cross Gram matrix between X and Y
        K_XY = exp.(-pairwise(Euclidean(), X, Y) ./ (2 * sigma^2))
        
        # Compute the MMD value
        mmd_value = mean(K_X) + mean(K_Y) - 2 * mean(K_XY)
        
        return mmd_value
    end
end 


# test the mmd function for two gaussian random variates

# using Random 
# using Statistics
# using LinearAlgebra

# # sample two samples of size 1000 from two gaussian distributions with different means and variances 
# Random.seed!(1234)

# # Different means but same variance

# X = randn(1000) .+ 1.0
# Y = randn(1000) .+ 2.0
# MMD(X, Y)


# # Different variances but same mean
# X = randn(1000) .+ 1.0
# Y = randn(1000) .* 2.0 .+ 1.0
# MMD(X, Y; kernel = ExponentialKernel())
# # Different means and different variances
# X = randn(1000) .+ 1.0
# Y = randn(1000) .* 0.01 .+ 100.0
# MMD(X, Y; kernel = ExponentialKernel())
