# imports
import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

from model import VAE, IWAE
from utils import AverageMeter

from IPython import embed

def elbo_loss(recon_image, image, mu, logvar,
              lambda_image=1.0, annealing_factor=1):
    """Bimodal ELBO loss function. 
    
    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.view(-1, 1 * 28 * 28), 
            image.view(-1, 1 * 28 * 28)), dim=1)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + annealing_factor * KLD)
    return ELBO

def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy
    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target 
            + torch.log(1 + torch.exp(-torch.abs(input))))



# Static parameters
batch_size = 100
n_epochs = 5
n_latents = 64
lr = 1e-3
log_interval = 100

# Load data
train_loader   = torch.utils.data.DataLoader(
    MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
N_mini_batches = len(train_loader)
test_loader    = torch.utils.data.DataLoader(
    MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

#model = VAE(n_latents)
model = IWAE(n_latents, n_particles=10)
optimizer = optim.Adam(model.parameters(), lr=lr)
#from IPython import embed
#embed()

def train(epoch):
    model.train()
    train_loss_meter = AverageMeter()

    # NOTE: is_paired is 1 if the example is paired
    for batch_idx, (image, label) in enumerate(train_loader):
        #recons, _ = model.reconstruct('iwae', image)
        #print(batch_idx)
        image      = Variable(image)
        image = image.permute(0, 2, 3, 1) # reshape to NHWC
        label       = Variable(label)
        batch_size = len(image)
        
        # refresh the optimizer
        optimizer.zero_grad()

        # pass data through model
        recon_image, z, mu, logvar = model(image)

        # compute ELBO 
        #train_loss = elbo_loss(recon_image, image, mu, logvar)
        train_loss = -model.elbo(image, recon_image, z, mu, logvar)
        train_loss_meter.update(train_loss.data, batch_size)
        
        # compute gradients and take step
        train_loss.backward()
        optimizer.step()

        
        #running_loss += train_loss.item()
        if batch_idx % 100 == 99:    # every 1000 mini-batches...
            #embed()
            # ...log the running loss
            #writer.add_scalar('training loss',
            #                running_loss / 1000,
            #                epoch * len(train_loader) + batch_idx)
            for name, param in model.named_parameters(): #model.parameters():
                #print(name)
                #print(param)
                writer.add_histogram(name, param, epoch) 
                writer.add_histogram(name + '/grad', param.grad, epoch)
            
        
        #if batch_idx % log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(image), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), train_loss_meter.avg))
                # ...log the running loss
    writer.add_scalar('training loss',
                    train_loss_meter.avg,
                    epoch)

    print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))


def test(epoch):
    model.eval()
    test_loss_meter = AverageMeter()

    for batch_idx, (image, label) in enumerate(test_loader):

        image = Variable(image)
        image = image.permute(0, 2, 3, 1) # reshape to NHWC
        label  = Variable(label)
        batch_size = len(image)

        recon_image, z, mu, logvar = model(image)

        #test_loss = elbo_loss(recon_image, image, mu, logvar)
        test_loss = -model.elbo(image, recon_image, z, mu, logvar)
        test_loss_meter.update(test_loss.data, batch_size)

    writer.add_scalar('test loss', test_loss_meter.avg, epoch)

    print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
    return test_loss_meter.avg

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/exp2')

# get some random training images
#dataiter = iter(train_loader)
#images, labels = dataiter.next()

# create grid of images
#img_grid = torchvision.utils.make_grid(images)

# show images
#matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
#writer.add_image('four_mnist_images', img_grid)

#writer.add_graph(model, images)
#writer.close()

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test_loss = test(epoch)
