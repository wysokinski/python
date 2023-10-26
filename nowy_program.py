#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:56:48 2023

@author: marcin
"""
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 5), -1)
g_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

observations_mean = tf.constant([np.mean(observed_values)], dtype=tf.float64)
mean_fn = lambda _: observations_mean
# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
    length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'))

gp = tfd.GaussianProcess(kernel, observed_index_points)

optimizer = tf.optimizers.Adam()

@tf.function
def optimize():
  with tf.GradientTape() as tape:
    loss = -gp.log_prob(observed_values)
  grads = tape.gradient(loss, gp.trainable_variables)
  optimizer.apply_gradients(zip(grads, gp.trainable_variables))
  return loss

for i in range(1000):
  neg_log_likelihood = optimize()
  if i % 100 == 0:
    print("Step {}: NLL = {}".format(i, neg_log_likelihood))
    print(gp.trainable_variables)
print("Final NLL = {}".format(neg_log_likelihood))
print(gp.kernel.amplitude)
fig, ax = plt.subplots(figsize=(6, 4.5))
# Plot bivariate distribution
samples = gp.sample(10)
# Plot samples
ax.plot(observed_index_points, observed_values, 'ro', alpha=.6,
        markeredgecolor='k', markeredgewidth=0.5)
ax.plot(observed_index_points, samples[[2]], 'bo', alpha=.6,
        markeredgecolor='k', markeredgewidth=0.5)
ax.plot(observed_index_points, samples[[9]], 'go', alpha=.6,
        markeredgecolor='k', markeredgewidth=0.5)
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$y $', fontsize=13)
ax.axis([-1, 1, -3, 3])
#ax.set_aspect('equal')
#ax.set_title('Samples from bivariate normal distribution')
#cbar = plt.colorbar(con)
#cbar.ax.set_ylabel('density: $p(y_1, y_2)$', fontsize=13)
plt.show()