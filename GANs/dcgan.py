"""
Deep Convolutional GANs

Training a GAN model to generate images using cifar10 as training dataset
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).


# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.


# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Defining the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),  # num_features = out_channels
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Defining the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)  # Flatten the result of the convolution


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Loading the dataset
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)  # We download the training set in the ./data folder and we apply the previous transformations on each image.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)  # We use dataLoader to get the images of the training set batch by batch.

    # Creating the generator
    net_generator = Generator()
    net_generator.cuda()
    net_generator.apply(weights_init)

    # Creating the discriminator
    net_discriminator = Discriminator()
    net_discriminator.cuda()
    net_discriminator.apply(weights_init)


    # Training the DCGANs
    criterion = nn.BCELoss()
    optimizer_discriminator = optim.Adam(net_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.99))
    optimizer_generator = optim.Adam(net_generator.parameters(), lr=0.0002, betas=(0.5, 0.99))

    for epoch in range(25):
        for index, data in enumerate(dataloader):  # data is a tuple (images batch, label)
            # 1st step: Updating the weights of the discriminator
            net_discriminator.zero_grad()  # Inizialize the weights to zero
            # Train the discriminator with real images
            real, _ = data
            input = Variable(real).to(device)
            target = Variable(torch.ones(input.size()[0])).to(device)
            output = net_discriminator(input)
            err_discriminator_real = criterion(output, target)

            # Train the discriminator with a fake image by the generator
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)  # 1st element is number of the batch
            fake = net_generator(noise)
            target = Variable(torch.zeros(input.size()[0])).to(device)
            output = net_discriminator(fake.detach())
            err_discriminator_fake = criterion(output, target)

            # Backpropagating the total error
            err_discriminator = err_discriminator_real + err_discriminator_fake
            err_discriminator.backward()
            optimizer_discriminator.step()

            # 2nd step: Updating the weights of the generator
            net_generator.zero_grad()
            target = Variable(torch.ones(input.size()[0])).to(device)
            output = net_discriminator(fake)
            err_generator = criterion(output, target)
            err_generator.backward()
            optimizer_generator.step()

            # 3rd step: Printing the losses and saving the real and the generated images
            print(
                f"[{epoch}/25][{index}/{len(dataloader)}] Loss_D: {err_discriminator.data:.4f} \
                Loss_G: {err_generator.data:.4f}"
            )
            if index % 100 == 0:
                vutils.save_image(real, "results/real_samples.png", normalize=True)
                fake = net_generator(noise)
                vutils.save_image(fake, f"results/fake_sample_epoch_{epoch}.png", normalize=True)


    # Save models
    torch.save(net_discriminator.state_dict(), "net_discriminator.pth")
    torch.save(net_generator.state_dict(), "net_generator.pth")

if __name__ == "__main__":
    main()