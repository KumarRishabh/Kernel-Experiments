import numpy as np 
import GeometricKernels
import matplotlib.pyplot as plt
import Manopt
import time

# Generate data
np.random.seed(0)
n = 100
X = np.random.rand(n, 2)
Y = np.random.rand(n, 2)


