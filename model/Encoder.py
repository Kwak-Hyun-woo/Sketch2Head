import numpy as np
import torch.nn as nn
import torch
from torchvision import models, transforms
from collections import OrderedDict

class Upsampling(nn.Module):
    def __init__(self, in_features, out_features, scale_factor = 2):
        super(Upsampling, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = 2*scale_factor
        self.stride = scale_factor
        self.pad_size = scale_factor // 2

        self.Transpose = nn.ConvTranspose2d(in_features, 
                                            out_features, 
                                            self.kernel_size, 
                                            stride= self.stride, 
                                            padding= self.pad_size)
        self.batch_norm = nn.BatchNorm2d(num_features=out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.Transpose(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Custom Squeeze module
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)
    
# Custom Reshape module
class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # consider batch size



class Encoder(nn.Module):
    def __init__(self, target = "ws", is_bezier = False):
        super(Encoder, self).__init__()

        # set output size fitting target size 
        if not is_bezier:
            # load pretrained resnet 18
            self.resnet_og = models.resnet18(pretrained=True)
            # remove last layer
            self.resnet = nn.Sequential(*list(self.resnet_og.children())[:-3])
            self.target = target
            if target == "z":
                # z size: torch.Size([1, 512])
                self.net = nn.Sequential(self.resnet,
                                        *list(self.resnet_og.children())[-3:-1],
                                        Reshape(1, 512),
                                        nn.Linear(512, 512)
                                        )
            elif target == "ws":
                # ws size: torch.Size([1, 14, 512])
                self.net = nn.Sequential(self.resnet,
                                        *list(self.resnet_og.children())[-3:-1],
                                        nn.Flatten(),
                                        nn.Linear(512, 7*512),
                                        nn.Linear(7*512, 14*512),
                                        Reshape(14, 512),    # fix to consider batch
                                        )
            elif target == "tri_plane":
                # tri_plane size: torch.Size([3, 96, 256, 256])
                self.net = nn.Sequential(self.resnet,   # [1, 256, 16, 16]
                                        # upsampling 
                                        Upsampling(256, 256, 2),    # [1, 256, 32, 32]
                                        Upsampling(256, 256, 2),    # [1, 256, 64, 64]
                                        Upsampling(256, 256, 2),    # [1, 256, 128, 128]
                                        Upsampling(256, 256, 2),    # [1, 256, 256, 256]
                                        # feature control 256 -> 3*96
                                        nn.Conv2d(256, 96*3, 3, stride=1, padding= 1), # [1, 96*3, 256, 256]
                                        Reshape(3, 96, 256, 256)    # fix to consider batch
                                        )
            else:
                raise Exception("set target arguments either z, w or tri_plane")
        else:   # bezier 
            self.net = nn.Sequential(
                                    Reshape(1024),   # consider batch 
                                    nn.Linear(2*512, 8*512),
                                    nn.BatchNorm1d(8*512),
                                    nn.ReLU(),
                                    nn.Linear(8*512, 14*512),
                                    nn.BatchNorm1d(14*512),
                                    nn.ReLU(),
                                    nn.Linear(14*512, 14*512),
                                    nn.BatchNorm1d(14*512),
                                    nn.ReLU(),
                                    Reshape(14, 512),    # fix to consider batch
                                    )


    def forward(self, x):
        x = self.net(x)
        return x
    
if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from dataset import FSPairedDataset
    from torchvision.transforms import InterpolationMode
    from PIL import Image

    # size = 256

    transform = transforms.Compose([
                # transforms.Resize(int(size * 1.12), InterpolationMode.BICUBIC),
                # transforms.RandomCrop(size),
                transforms.ToTensor()])
    
    dataset = FSPairedDataset('../data', None, transform = transform)
    input_dict = dataset[0]
    input = input_dict["img"].unsqueeze(dim=0)

    print(f"input size: {input.shape}")
    net = Encoder(target="tri_plane")
    output = net(input)
    print(f"output size: {output.shape}")