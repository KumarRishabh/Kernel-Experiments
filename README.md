# Kernel-Experiments
A sandbox for experiments using Geometric Kernels and its application on two-sample tests.

# Some preliminary plans for experiments

A few papers have popped up since the seminal paper by Alex Smola and group released their paper on using Kernels methods for two sample texts [[base-reference]], namely [[ref1]] and [[ref2]] (these references were sent to the Cyrus group). The recent result by [[Cyrus-et-al]] introduces Kernels on SPD manifolds, which promises improved power and computational complexity as compared to the previous iterations of the kernel method of the two sample test. The above claim warrants further testing, which involves testing the kernel on the following datasets: 

- [x] Change the colors of the computational complexity plot (the color scheme of the geometric kernels and Monte Carlo kernels should differ across classes and should be consistent within)
- [x] Get the Chebyshev formula computation right. In turn, the Cauchy kernel computation would be accurate and right. 
- [ ] Add the Suvrit Sra (unnormalized stein kernel) computation right and add it to the table. Add a footnote for when the unnormalized version is normalized, it transforms to the Beta-prime kernel. 
- [ ] Add a remark after the Beta-prime example that we do not use the entire specturm for the computation of Beta-prime kernel (why not? Is computation of determinant cheaper?). Hence, Beta-prime computation takes less time. 
- [ ] (Optional) Test spectrum of gram matrices for Cauchy and Beta-prime kernel. Through some theory of the expressivity of kernels, equivalence of kernels to Gaussian Process regression, the spectral distribution gives some information about expressivity (Thanks to the suggestion by Dino Sejdinovic.)
- [ ] Package all of the code coherently for the purposes of publication. 
- [ ] Write down the mechanism by which the covariance matrices are estimated from an EEG-BCI signal. 
- [ ] Add the Cauchy kernel test to 2-sample test for Franziskus's experiment. 

