# Structured Sparse Dictionary Selection

This is an implementation of the Structured Sparse Dicationary Selection (SSDS) algorithm described in

<pre>Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
Representative Selection with Structured Sparsity,
Pattern Recognition, vol. 63, pp. 268-278, 2017,
http://dx.doi.org/10.1016/j.patcog.2016.10.014.</pre>

The code is provided for research purposes only and without any warranty. Please cite this paper if you use the code as part of your work.

# Environment

The code is tested on 64-bit Windows with MATLAB R2015a, coming with pre-compiled mex files via Visual Studio 2013.

# Usage

1. Download the Eigen library (versoion 3) from http://eigen.tuxfamily.org, and put the folder "eigen3" in the "SSDS" directory.
2. Compile related functions in the folder "SSDS" based on your architecture. You can run compile.m to complete the compilation in the aforementioned enviroment. 
3. Input the data matrix X to demo_SSDS.m, and run the m-file for Structured Sparse Dictionary Selection.

# References

[1] Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
Representative Selection with Structured Sparsity,
Pattern Recognition, vol. 63, pp. 268-278, 2017.

[2] G. Gallo, M.D. Grigoriadis, R.E. Tarjan, A fast parametric maximum flow algorithm
and applications, SIAM Journal on Computing, vol. 18, no.1, 30-55, 1989.

[3] M. Babenko, J. Derryberry, A. Goldberg, R. Tarjan, Y. Zhou, Experimental
evaluation of parametric max-flow algorithms, in C. Demetrescu (Ed.)
Experimental Algorithms, Springer, Berlin, Heidelberg, pp. 256–269, 2007.

[4] Y. Cong, J. Yuan, J. Liu, Sparse reconstruction cost for abnormal event detection,
in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 
pp. 3449–3456, 2011.

[5] E. Elhamifar, G. Sapiro, R. Vidal, See all by looking at a few: Sparse modeling for
finding representative objects, in Proceedings of IEEE Conference on Computer
Vision and Pattern Recognition, pp. 1600–1607, 2012.

[6] K. Nagano, Y. Kawahara, Structured convex optimization under submodular
constraints, in Proceedings of Conference on Uncertainty in Artificial Intelligence, 
pp. 459–468, 2013.
