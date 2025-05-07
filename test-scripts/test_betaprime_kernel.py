import torch
import unittest
from experiments.BetaprimeKernel import BetaprimeKernel


class TestBetaprimeKernel(unittest.TestCase):
    def test_identity_matrix(self):
        d = 4
        I = torch.eye(d, dtype=torch.float64)
        kernel = BetaprimeKernel()
        val = kernel.kernel(I, I)
        # For identity matrices, trace = d, norm^2 = d so (d^2)/(d*d) = 1
        self.assertAlmostEqual(val.item(), 1.0, places=6)

    def test_diag_matrices(self):
        A = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        B = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
        kernel = BetaprimeKernel()
        # Manually: trace = 1*3 + 2*4 = 11, normsq_A = 1+4 = 5, normsq_B = 9+16 = 25
        expected = 121.0 / (5.0 * 25.0)
        val = kernel.kernel(A, B).item()
        self.assertAlmostEqual(val, expected, places=6)

    def test_pairwise_batch(self):
        A = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        B = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
        kernel = BetaprimeKernel()
        X = torch.stack([A, B])  # shape (2,2,2)
        G = kernel.pairwise(X, X)
        # G should be symmetric and diagonal entries = 1
        self.assertEqual(G.shape, (2, 2))
        self.assertAlmostEqual(G[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(G[1, 1].item(), 1.0, places=6)
        # off-diagonal equals kernel(A,B)
        off = G[0, 1].item()
        expected = kernel.kernel(A, B).item()
        self.assertAlmostEqual(off, expected, places=6)


if __name__ == '__main__':
    unittest.main()