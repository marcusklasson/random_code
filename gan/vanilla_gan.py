
"""
https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
Requires Tensorflow 1.
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

# Data
X = tf.placeholder(tf.float32, shape=[None, 784])

"""
Define weights and biases (parameters) for the discriminator network D
Output of D has size 1, since it represents the probability of an 
image to be real or fake. 
""" 
D_W1 = tf.Variable(tf.random_normal(shape=[784, 128], mean=0.0, stddev=0.01))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(tf.random_normal(shape=[128, 1], mean=0.0, stddev=0.01))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator network G
Z = tf.placeholder(tf.float32, [None, 100])

G_W1 = tf.Variable(tf.random_normal(shape=[100, 128], mean=0.0, stddev=0.01))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(tf.random_normal(shape=[128, 784], mean=0.0, stddev=0.01))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
	return np.random.uniform(-1, 1, size=[m, n])

def generator(z):
	G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
	G_logits = tf.matmul(G_h1, G_W2) + G_b2
	return tf.nn.sigmoid(G_logits)

def discriminator(x):
	D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
	D_logit = tf.matmul(D_h1, D_W2) + D_b2
	D_prob = tf.nn.sigmoid(D_logit)
	return D_prob, D_logit

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    samples = (samples > 0.5).astype(np.float32)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')
    return fig

G_sample = generator(Z)
D_real, D_real_logit = discriminator(X)
D_fake, D_fake_logit = discriminator(G_sample)

# Loss functions 
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

batch_size = 128
Z_dim = 100
print_every = 10000
n_iter = 100000
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
	os.makedirs('out/')

for it in range(n_iter):
    if it % print_every == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/iter_{}.png'.format(it), bbox_inches='tight')
        plt.close(fig)

    X_batch, _ = mnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_batch, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    if it % print_every == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()