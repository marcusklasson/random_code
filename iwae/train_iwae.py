
import os
import sys
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vae_iwae import VAE, IWAE

from IPython import embed

def train_step(model, image, optimizer):
    batch_size = image.shape[0]
    n_particles = model.n_particles

    with tf.GradientTape() as tape:
        #recon_image, z, mu, logvar = model(image)
        recon_image, z, mu, logvar = model.forward_pass(image)
        elbo = model.elbo(tf.reshape(image, [batch_size, 784]), tf.reshape(recon_image, [n_particles, batch_size, 784]), z, mu, logvar)
        train_loss = -elbo

    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss

def save_checkpoint(manager, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    manager.save()

def load_checkpoint(ckpt, folder='./'):
    status = ckpt.restore(tf.train.latest_checkpoint(folder)).expect_partial()
    print("Loaded checkpoint {}".format(tf.train.latest_checkpoint(folder)))
    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=25, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--n-samples', type=int, default=16, metavar='N',
                        help='number of samples frawn from prior [default: 16]')

    args = parser.parse_args()

    # Load MNIST dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    n_train = train_images.shape[0]
    n_test = test_images.shape[0]
    n_batches_train = int(n_train / args.batch_size)
    n_batches_test = int(n_test / args.batch_size)

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    # Create batches and shuffle dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(n_train).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(n_test).batch(args.batch_size)

    # Create model and optimizer
    #model = VAE(args.n_latents, n_particles=1)
    model = IWAE(args.n_latents, n_particles=10)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    # Create checkpoint manager for saving models
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    #manager = tf.train.CheckpointManager(ckpt, './trained_models', max_to_keep=3)
    #status = load_checkpoint(ckpt, './trained_models')
    
    best_loss = float("inf")
    print("Start training...")
    for epoch in range(1, args.epochs+1):
        
        train_loss = 0.
        samples = model.sample(args.n_samples)
        for batch_idx, image in enumerate(train_dataset):
            loss = train_step(model, image, optimizer)
            train_loss += loss

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss / n_batches_train))
        
        test_loss = 0.
        for batch_idx, image in enumerate(test_dataset):
            loss = train_step(model, image, optimizer)
            test_loss += loss
            
        print('\tTest Loss: {:.4f}'.format(test_loss / n_batches_test))
        
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        #save_checkpoint(manager, is_best, folder='./trained_models')
        
        # Sample from prior
        #samples, _ = model.reconstruct(sampling='iwae', data=image[:16])
        samples = model.sample(args.n_samples)
        sz = np.sqrt(samples.shape[0])
        fig = plt.figure(figsize=(sz,sz))
        for i in range(samples.shape[0]):
            plt.subplot(sz, sz, i+1)
            plt.imshow(samples[i, :, :, 0])
            plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('sample_image_epoch{}.png'.format(epoch))
        