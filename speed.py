#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import numpy as np
from time import time

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 4096
A, B = np.random.random((size*1000, 3)).astype(np.float64), np.random.random((size*1000, 3)).astype(np.float64)
Y, Z = np.random.random((size*1000, 3)).astype(np.float64), np.random.random((size*1000, 3)).astype(np.float64)
W, X = np.random.random((size*100, 3)).astype(np.float32), np.random.random((size*100, 3)).astype(np.float32)
J, K = np.random.random((size*100, 3)).astype(np.float32), np.random.random((size*100, 3)).astype(np.float32)
H1, H2 = np.random.random((size*100, 3)).astype(np.float16), np.random.random((size*100, 3)).astype(np.float16)
H3, H4 = np.random.random((size*100, 3)).astype(np.float16), np.random.random((size*100, 3)).astype(np.float16)
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))



# Sum Range
N = 100
t = time()
for i in range(N):
   a = np.sum(A*B, axis=1)
delta = time() - t
print(a)
print('Np.sum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))

# einsum
N = 100
t = time()
for i in range(N):
    a = np.einsum('ij, ij->i', Y, Z)
delta = time() - t
print(a)
print('einsum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))


# Sum Range
N = 100
t = time()
for i in range(N):
   a = np.sum(W*X, axis=1)
delta = time() - t
print(a)
print('Np.sum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))

# einsum
N = 100
t = time()
for i in range(N):
    a = np.einsum('ij, ij->i', J, K)
delta = time() - t
print(a)
print('einsum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))


# Sum Range
N = 100
t = time()
for i in range(N):
   a = np.sum(H1*H2, axis=1)
delta = time() - t
print(a)
print('Np.sum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))

# einsum
N = 100
t = time()
for i in range(N):
    a = np.einsum('ij, ij->i', H3, H4)
delta = time() - t
print(a)
print('einsum two %dx%d matrices in %0.2f ms.' % (size, size, 1e3 * delta / N))


breakpoint()



# Matrix multiplication
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
del A, B

# Vector multiplication
N = 5000
t = time()
for i in range(N):
    np.dot(C, D)
delta = time() - t
print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
del C, D

# Singular Value Decomposition (SVD)
N = 3
t = time()
for i in range(N):
    np.linalg.svd(E, full_matrices = False)
delta = time() - t
print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
del E

# Cholesky Decomposition
N = 3
t = time()
for i in range(N):
    np.linalg.cholesky(F)
delta = time() - t
print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

# Eigendecomposition
t = time()
for i in range(N):
    np.linalg.eig(G)
delta = time() - t
print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))