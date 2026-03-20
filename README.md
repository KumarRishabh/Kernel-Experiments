# Kernel-Experiments
A sandbox for experiments using Geometric Kernels and its application on two-sample tests.

# Some preliminary plans for experiments

A few papers have popped up since the seminal paper by Alex Smola and group released their paper on using Kernels methods for two sample texts [[base-reference]], namely [[ref1]] and [[ref2]] (these references were sent to the Cyrus group). The recent result by [[Cyrus-et-al]] introduces Kernels on SPD manifolds, which promises improved power and computational complexity as compared to the previous iterations of the kernel method of the two sample test. The above claim warrants further testing, which involves testing the kernel on the following datasets: 

- [ ] Change the colors of the computational complexity plot (the color scheme of the geometric kernels and monte carlo kernels should differ across classes and should be consistent within)
- [ ] Get the chebyshev formula computation right. In turn, the cauchy kernel computation would be accurate and right. 
- [ ] Add the Suvrit Sra (unnormalized stein kernel) computation right and add it to the table. Add a footnote for when the unnormalized version is normalized, it transforms to the betaprime kernel. 
- [ ] Add a remark after the Betaprime example that we do not use the entire specturm for the computation of betaprime kernel (why not? Is computation of determinant cheaper?). Hence, betaprime computation takes less time. 
- [ ] (Optional) Test spectrum of gram matrices for cauchy and betaprime kernel. Through some theory of the expressivity of kernels, equivalence of kernels to Gaussian Process regression, the spectral distribution gives some information about expressivity (Thanks to the suggestion by Dino Sedjinovic.)
- [ ] Package all of the code coherently for the purposes of publication. 
- [ ] Add the cauchy kernel test to 2-sample test for Franziskus's experiment. 

