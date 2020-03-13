
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

class VAE(tf.keras.Model):
#class VAE(object):
    def __init__(self, n_latents, n_particles=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_latents)
        self.decoder = Decoder(n_latents)
        self.n_latents = n_latents
        self.n_particles = n_particles
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(n_latents), scale=1),
                        reinterpreted_batch_ndims=1)


    def reparameterize(self, mu, logvar):
        batch_size, n_latents = logvar.shape
        n_particles = self.n_particles
        eps = tf.random.normal(shape=[n_particles, batch_size, n_latents], mean=0.0, stddev=1.0)
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(logvar)), eps)) #uses broadcasting, z=[n_parts, n_batches, n_z]
        return z

    def encode(self, image):
        mu, logvar = self.encoder(image)
        return mu, logvar

    def decode(self, z):
        batch_size = z.shape[1]
        z = tf.reshape(z, [self.n_particles*batch_size, self.n_latents])
        logits = self.decoder(z)
        logits = tf.reshape(logits, [self.n_particles, batch_size, 28, 28, 1])
        return logits

    def sample(self, n_samples):
        n_latents = self.n_latents
        prior_samples = self.prior.sample(n_samples)
        #prior_samples = tf.expand_dims(prior_samples, axis=0)
        img_samples = tf.nn.sigmoid(self.decoder(prior_samples))
        #img_samples = tf.squeeze(img_samples, axis=0)
        return img_samples

    def _log_p_z(self, z):
        '''
        Log of normal distribution with zero mean and one var
        z is [n_particles, batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # term1 = 0
        term2 = self.n_latents * tf.math.log(2*np.pi)
        term3 = tf.reduce_sum(tf.square(z), 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term2 + term3
        log_p_z = -.5 * all_

        return log_p_z

    def _log_q_z_given_x(self, z, mean, log_var):
        '''
        Log of normal distribution
        z is [n_particles, batch_size, n_z]
        mean is [batch_size, n_z]
        log_var is [batch_size, n_z]
        output is [n_particles, batch_size]
        '''

        # term1 = tf.log(tf.reduce_prod(tf.exp(log_var_sq), reduction_indices=1))
        term1 = tf.reduce_sum(log_var, axis=1) #sum over dimensions n_z so now its [batch]

        term2 = self.n_latents * tf.math.log(2*np.pi)
        dif = tf.square(z - mean)
        dif_cov = dif / tf.exp(log_var)
        # term3 = tf.reduce_sum(dif_cov * dif, 1) 
        term3 = tf.reduce_sum(dif_cov, 2) #sum over dimensions n_z so now its [particles, batch]

        all_ = term1 + term2 + term3
        log_p_z_given_x = -.5 * all_

        return log_p_z_given_x

    def _binary_cross_entropy_with_logits(self, target, pred_no_sig):
        """Sigmoid Activation + Binary Cross Entropy
        @param pred_no_sig: torch.Tensor (size N)
        @param target: torch.Tensor (size N)
        @return loss: torch.Tensor (size N)
        """
        #if not (target.shape == pred_no_sig.shape):
        #    raise ValueError("Target size ({}) must be the same as input size ({})".format(
        #        target.shape, pred_no_sig.shape))

        #return tf.reduce_sum((tf.nn.relu(pred_no_sig)  - pred_no_sig * target 
        #        + tf.math.log(1 + tf.math.exp(-tf.math.abs(pred_no_sig)) ) ), axis=2)

        
        reconstr_loss = \
                tf.reduce_sum(tf.maximum(pred_no_sig, 0) 
                                - pred_no_sig * target 
                                + tf.math.log(1 + tf.exp(-tf.abs(pred_no_sig))), 2) #sum over dimensions
        return -reconstr_loss
        

    def elbo(self, x, x_recon, z, mean, log_var):

        elbo = self._binary_cross_entropy_with_logits(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        elbo = tf.reduce_mean(elbo, 1) #average over batch
        elbo = tf.reduce_mean(elbo) #average over particles

        return elbo

    def forward_pass(self, image):
        mu, logvar = self.encode(image)
        # reparametrization trick to sample
        z          = self.reparameterize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.decode(z)
        return img_recon, z, mu, logvar

    def reconstruct(self, sampling, data):

        # #Ramdomly select a batch
        # batch = []
        # while len(batch) != self.batch_size:
        #     datapoint = data[np.random.randint(0,len(data))]
        #     batch.append(datapoint)
        batch = data
        batch_size = batch.shape[0]

        if sampling == 'vae':

            #Encode and get p and q
            recons, z, mu, logvar = self.forward_pass(batch)
            # Flatten the images
            batch = tf.reshape(batch, [batch_size, 784])
            recons = tf.reshape(recons, [self.n_particles, batch_size, 784])
            log_ws = self._binary_cross_entropy_with_logits(batch, recons) + self._log_p_z(z) - self._log_q_z_given_x(z, mu, logvar)
            #log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

            # print log_ws.shape
            # print recons.shape

            return recons, batch

        if sampling == 'iwae':

            recons_resampled = []
            for i in range(self.n_particles):

                #Encode and get p and q.. log_ws [K,B,1], reons [K,B,X]
                """
                mu, logvar = self.encode(batch)
                # reparametrization trick to sample
                z          = self.reparameterize(mu, logvar)
                # reconstruct inputs based on that gaussian
                recons  = self.decode(z)
                """
                recons, z, mu, logvar = self.forward_pass(batch)
                # Flatten the images
                batch = tf.reshape(batch, [batch_size, 784])
                recons = tf.reshape(recons, [self.n_particles, batch_size, 784])
                log_ws = self._binary_cross_entropy_with_logits(batch, recons) + self._log_p_z(z) - self._log_q_z_given_x(z, mu, logvar)
                #log_ws, recons = self.sess.run((self.log_w, self.x_reconstr_mean), feed_dict={self.x: batch})

                #log normalize
                max_ = np.max(log_ws, axis=0)
                lse = np.log(np.sum(np.exp(log_ws-max_), axis=0)) + max_
                log_norm_ws = log_ws - lse

                # ws = np.exp(log_ws)
                # sums = np.sum(ws, axis=0)
                # norm_ws = ws / sums


                # print log_ws
                # print
                # print lse
                # print
                # print log_norm_ws
                # print 
                # print np.exp(log_norm_ws)
                # fsdfa

                #sample one based on cat(w)

                samps = []
                for j in range(batch_size):

                    samp = np.argmax(np.random.multinomial(1, np.exp(log_norm_ws.T[j])-.000001))
                    samps.append(recons[samp][j])
                    # print samp

                # print samps
                # print samps.shape
                # fasdf
                recons_resampled.append(np.array(samps))

            recons_resampled = np.array(recons_resampled)
            # print recons_resampled.shape


            return recons_resampled, batch
    """
    def call(self, image):
        mu, logvar = self.encode(image)
        # reparametrization trick to sample
        z          = self.reparameterize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.decode(z)
        return img_recon, z, mu, logvar
    """

class IWAE(VAE):

    def elbo(self, x, x_recon, z, mean, log_var):

        # [P, B]
        temp_elbo = self._binary_cross_entropy_with_logits(x, x_recon) + self._log_p_z(z) - self._log_q_z_given_x(z, mean, log_var)

        max_ = tf.reduce_max(temp_elbo, axis=0) #over particles? so its [B]

        elbo = tf.math.log(tf.reduce_mean(tf.exp(temp_elbo-max_), 0)) + max_  #mean over particles so its [B]

        elbo = tf.reduce_mean(elbo) #over batch

        return elbo


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