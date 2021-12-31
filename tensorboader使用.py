import torchvision
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./log")

r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*3,
                                    'xcosx':i*10,
                                    'tanx': i*20}, i)
writer.close()
