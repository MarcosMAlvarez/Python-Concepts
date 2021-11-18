"""
Generating images from a trained Generator
"""
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

from dcgan import Generator

generator = Generator()
generator.load_state_dict(torch.load("net_generator.pth"))
generator.eval()

noise = Variable(torch.randn(64, 100, 1, 1))
images = generator(noise)
vutils.save_image(images[0], "test_load_model.png", normalize=True)
