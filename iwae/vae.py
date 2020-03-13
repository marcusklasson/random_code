
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

class VAE(tf.keras.Model):
    def __init__(self, n_latents):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_latents)
        self.decoder = Decoder(n_latents)
        self.n_latents = n_latents
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(n_latents), scale=1),
                        reinterpreted_batch_ndims=1)

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=logvar.shape)
        return eps * tf.exp(logvar * .5) + mu

    def encode(self, image):
        mu, logvar = self.encoder(image)
        return mu, logvar

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def sample(self, n_samples):
        n_latents = self.n_latents
        prior_samples = self.prior.sample(n_samples)
        img_samples = tf.nn.sigmoid(self.decode(prior_samples))
        return img_samples

    def call(self, image):
        mu, logvar = self.encode(image)
        # reparametrization trick to sample
        z          = self.reparameterize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.decode(z)
        return img_recon, mu, logvar

class Encoder(tf.keras.Model):
    """Parametrizes q(z|x).
    @param latent_dim: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(Encoder, self).__init__()
        self.fc1   = layers.Dense(units=512, activation=None) #nn.Linear(784, 512)
        self.fc2   = layers.Dense(units=512, activation=None) #nn.Linear(512, 512)
        self.fc3  = layers.Dense(units=n_latents * 2, activation=None) #nn.Linear(512, latent_dim)
        self.n_latents = n_latents

    def call(self, x):
        n_latents = self.n_latents

        h = tf.nn.leaky_relu(self.fc1(tf.reshape(x, [-1, 784])))
        h = tf.nn.leaky_relu(self.fc2(h))
        out = self.fc3(h)
        return out[:, :n_latents], out[:, n_latents:]

class Decoder(tf.keras.Model):
    """Parametrizes p(x|z).
    @param latent_dim: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(Decoder, self).__init__()
        self.fc1   = tf.keras.layers.Dense(units=512, activation=None) #nn.Linear(n_latents, 512)
        self.fc2   = tf.keras.layers.Dense(units=512, activation=None) #nn.Linear(512, 512)
        self.fc3   = tf.keras.layers.Dense(units=784, activation=None) #nn.Linear(512, 512)
        #self.swish = Swish()

    def call(self, z):
        h = tf.nn.leaky_relu(self.fc1(z))
        h = tf.nn.leaky_relu(self.fc2(h))
        out = tf.reshape(self.fc3(h), [-1, 28, 28, 1])
        return out  # NOTE: no sigmoid here. 