import tensorflow as tf
import torch

from PIL import Image

import yapl

class SimpleTorch:
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        target = self.targets[item]

        return {
            "image" : torch.tensor(img, dtype=torch.float),
            'target' : torch.tensor(target, dtype=torch.long)
        }

def getImageloader(image_paths, targets):
    if yapl.backend == 'torch':
        return SimpleTorch(
            images=image_paths,
            targets=targets
        )
    elif yapl.backend == 'tf':
        pass
    
    else:
        pass
