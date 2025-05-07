import torch
import seaborn as sns

class BetaprimeKernel:
    """
    Beta-prime kernel for SPD matrices on the manifold.
    
    The kernel is defined as:
    
        k(X, Y) = (det(X)*det(Y) / (det(X + Y)**2))**alpha
    
    X and Y are assumed to be SPD matrices.
    """

    def __init__(self, alpha: float = 1.0):
        """
        alpha: Exponent for the kernel. > n - 1, where n is the dimension of the SPD matrices.
        """
        self.alpha = alpha

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the betaprime kernel between two SPD matrices X and Y.
        
        Args:
            X (torch.Tensor): A symmetric positive definite matrix.
            Y (torch.Tensor): A symmetric positive definite matrix.
            
        Returns:
            torch.Tensor: Scalar value of the betaprime kernel.
        """
        det_X = torch.det(X)
        det_Y = torch.det(Y)
        det_sum = torch.det(X + Y)
        
        # Small epsilon to avoid division by zero issues.
        eps = 1e-12
        kernel_val = ((det_X * det_Y) / (det_sum**2 + eps)) ** self.alpha
        return kernel_val
    
    def pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise betaprime kernel between two batches of SPD matrices.
        
        Args:
            X (torch.Tensor): Batch of SPD matrices of shape (n, d, d).
            Y (torch.Tensor): Batch of SPD matrices of shape (m, d, d).
            
        Returns:
            torch.Tensor: Pairwise kernel values of shape (n, m).
        """
        n = X.shape[0]
        m = Y.shape[0]
        K = torch.zeros((n, m), dtype=X.dtype, device=X.device)
        
        for i in range(n):
            for j in range(i+1, m):
                K[i, j] = self(X[i], Y[j])
                K[j, i] = K[i, j]
        
        # Diagonal entries
        for i in range(n):
            K[i, i] = self(X[i], X[i])
        return K
    
# Example usage
if __name__ == "__main__":
    # From a sample of 20 SPD matrices of size 2x2
    d = 2
    n_samples = 2000
    X = torch.randn(n_samples, d, d)
    Y = torch.randn(n_samples, d, d)
    X = torch.bmm(X, X.transpose(1, 2)) + 0.1 * torch.eye(d).unsqueeze(0)
    Y = torch.bmm(Y, Y.transpose(1, 2)) + 0.1 * torch.eye(d).unsqueeze(0)
    kernel = BetaprimeKernel(alpha=1.0)
    K = kernel(X, Y)
    print("Kernel value:", K)
    # print("Pairwise kernel matrix:", kernel.pairwise(X, Y))
    # Compute the pairwise kernel matrix
    K_matrix = kernel.pairwise(X, X)  # Using same matrices for demo

    # Pretty print using matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    sns.heatmap(K_matrix.detach().numpy(), cmap="viridis", annot=True, fmt=".4f")
    plt.title("Beta-prime Kernel Matrix")
    plt.tight_layout()
    plt.show()

    # For terminal display
    print("Kernel matrix shape:", K_matrix.shape)
    torch.set_printoptions(precision=4, sci_mode=False)
    print("Kernel matrix:\n", K_matrix)
