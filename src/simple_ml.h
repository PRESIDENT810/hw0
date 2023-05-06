#ifndef HW0_SIMPLE_ML_H
#define HW0_SIMPLE_ML_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    unsigned long iters = m / batch;
    assert(m%batch == 0);
    // Z = X_batch @ theta: (batch, k)
    float *Z = new float[batch * k];
    float *Iy = new float[batch * k];
    float *gradient = new float[n * k];
    for (unsigned long iter = 0; iter < iters; iter++) {
        const float *X_batch = X + iter * batch * n;
        const unsigned char *y_batch = y + iter * batch;
        // X: (batch, n); y: (batch, ); theta: (n, k)
        // Construct Z
        for (int k_i = 0; k_i < k; k_i++) {
            for (int batch_i = 0; batch_i < batch; batch_i++) {
                Z[batch_i * k + k_i] = 0.0;
                for (int n_i = 0; n_i < n; n_i++) {
                    Z[batch_i * k + k_i] += X_batch[batch_i * n + n_i] * theta[n_i * k + k_i];
                }
            }
        }
        for (int batch_i = 0; batch_i < batch; batch_i++) {
            float row_sum = 0;
            for (int k_i = 0; k_i < k; k_i++) {
                Z[batch_i * k + k_i] = exp(Z[batch_i * k + k_i]);
                row_sum += Z[batch_i * k + k_i];
            }
            for (int k_i = 0; k_i < k; k_i++) {
                Z[batch_i * k + k_i] /= row_sum;
            }
        }
        // Construct Iy
//        memset(Iy, 0, batch*k);
        for (int batch_i = 0; batch_i < batch; batch_i++) {
            for (int k_i = 0; k_i < k; k_i++) {
                Iy[batch_i * k + k_i] = 0.0;
            }
        }
        for (int batch_i = 0; batch_i < batch; batch_i++) {
            int idx = y_batch[batch_i];
            Iy[batch_i * k + idx] = 1;
        }
        // Construct gradient
        for (int batch_i = 0; batch_i < batch; batch_i++) {
            for (int k_i = 0; k_i < k; k_i++) {
                Z[batch_i * k + k_i] -= Iy[batch_i * k + k_i];
            }
        }
        for (int k_i = 0; k_i < k; k_i++) {
            for (int n_i = 0; n_i < n; n_i++) {
                gradient[n_i * k + k_i] = 0.0;
                for (int batch_i = 0; batch_i < batch; batch_i++) {
                    gradient[n_i * k + k_i] += X_batch[batch_i * n + n_i] * Z[batch_i * k + k_i];
//                    printf("X_batch[%d*%d+%d]=%f, Z[%d*%d+%d]=%f, gradident[%d*%d+%d]=%f\n",
//                           batch_i, n, n_i, X_batch[batch_i*n+n_i],
//                           batch_i, k, k_i, Z[batch_i * k + k_i],
//                           n_i, k, k_i, gradient[n_i * k + k_i]);
                }
                gradient[n_i * k + k_i] /= float(batch);
            }
        }
        for (int n_i = 0; n_i < n; n_i++) {
            for (int k_i = 0; k_i < k; k_i++) {
                theta[n_i * k + k_i] -= lr * gradient[n_i * k + k_i];
            }
        }

    }
    delete[]Z;
    delete[]Iy;
    delete[]gradient;
}

#endif //HW0_SIMPLE_ML_H
