#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <tqdm/tqdm.h>
#include <cmath>
#include <omp.h>                  // for OpenMP pragmas

namespace py = pybind11;
using Matrix = Eigen::MatrixXd;

// A small inlined function to compute determinant safely.
// We take max(determinant, 1e-12) to avoid returning zero.
inline double compute_det_safe(const Matrix& A) {
    double detA = A.determinant();
    return (detA > 1e-12 ? detA : 1e-12);
}

py::array_t<double> pairwise_kernel(py::array_t<double> x_in,
                                    py::array_t<double> y_in,
                                    double alpha) {
    // Expect contiguous row-major numpy arrays of shape (n, d, d) and (m, d, d)
    // | x_in | : shape (n, d, d)  => total length = n*d*d
    // | y_in | : shape (m, d, d)  => total length = m*d*d
    //
    // We will create a (n x m) output array and fill it in parallel.

    // 1. Get unchecked buffers for raw pointers
    auto X = x_in.unchecked<3>();
    auto Y = y_in.unchecked<3>();

    int n = static_cast<int>(X.shape(0));
    int m = static_cast<int>(Y.shape(0));
    int d = static_cast<int>(X.shape(1));

    // Validate that shapes are consistent: X.shape() == (n, d, d), Y.shape() == (m, d, d)
    if (X.shape(1) != X.shape(2) || Y.shape(1) != Y.shape(2) || X.shape(1) != Y.shape(1)) {
        throw std::runtime_error("Input arrays must be shape (n, d, d) and (m, d, d) with the same d.");
    }

    // 2. Allocate output (n x m) and get a raw pointer to its data
    py::array_t<double> K({n, m});
    auto K_buf = K.mutable_unchecked<2>();
    double* K_data_ptr = K_buf.mutable_data(0, 0);  
    // Access via K_data_ptr + (i*m + j) for element (i, j).

    // 3. Raw data pointers for X and Y:
    //    Each block X(i,:,:) begins at offset i * (d*d), same for Y.
    const double* X_data_ptr = x_in.data();  // pointer to first element of entire array
    const double* Y_data_ptr = y_in.data();

    // // 4. Set up a C++ progress bar with total = n rows
    // auto prog = tqdm::range(n);
    // prog.set_description("Computing pairwise kernel");

    // 5. Parallel loop over i using OpenMP
    #pragma omp parallel
    {
        // Each thread will have its own local det_X and Xi_map.
        Matrix Xi_local(d, d);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            // 5.1  Map X(i,:,:) into an Eigen::Map to avoid copying
            const double* Xi_ptr = X_data_ptr + static_cast<std::size_t>(i) * d * d;
            Eigen::Map<const Matrix> Xi_map(Xi_ptr, d, d);

            // 5.2 Compute det(Xi) once for this row
            double det_X = compute_det_safe(Xi_map);

            // 5.3 Now iterate over all j in this thread
            for (int j = 0; j < m; ++j) {
                // 5.3.1 Map Y(j,:,:) into Eigen::Map
                const double* Yj_ptr = Y_data_ptr + static_cast<std::size_t>(j) * d * d;
                Eigen::Map<const Matrix> Yj_map(Yj_ptr, d, d);

                // 5.3.2 Compute det(Yj)
                double det_Y = compute_det_safe(Yj_map);

                // 5.3.3 Compute Xi + Yj into a temporary matrix (reuse Xi_local)
                Xi_local = Xi_map + Yj_map;  
                double det_sum = compute_det_safe(Xi_local);

                // 5.3.4 Compute the kernel entry
                double val = std::pow((det_X * det_Y) / (det_sum * det_sum + 1e-12), alpha);

                // 5.3.5 Write into K(i, j). Because multiple threads write to different rows,
                //           no data race occurs for K_data_ptr + (i*m + j).
                K_data_ptr[static_cast<std::size_t>(i) * m + j] = val;
            }

            // 5.4 After finishing row i, update progress bar
            // #pragma omp critical
            // {
            //     prog.update(1);
            // }
        }
    } // end of parallel region

    return K;
}

PYBIND11_MODULE(betaprime_cpp, m) {
    m.def("pairwise_kernel", &pairwise_kernel,
          "Compute BetaPrime kernel matrix (optimized C++ version)",
          py::arg("X"), py::arg("Y"), py::arg("alpha") = 1.0);
}