import torch
import torch.nn as nn
import torch.nn.functional as F
import atariari.methods.encoders as encoders
from a2c_ppo_acktr.utils import init



class AdjustableNatureCNN(nn.Module):  

    def __init__(self, input_channels, args, receptive_field=16):
        super().__init__()

        if args.receptive_field == 16:
            self.strides = [4, 2, 2, 1]
            self.final_conv_size = 64 * 9 * 6
            self.final_conv_shape = (64, 9, 6)
        elif args.receptive_field == 8:
            self.strides = [4, 2, 1, 2]
            self.final_conv_size = 64 * 10 * 7
            self.final_conv_shape = (64, 10, 7)
        elif args.receptive_field == 4:
            self.strides =  [4, 1, 1, 4]
            self.final_conv_size = 64 * 11 * 8
            self.final_conv_shape = (64, 11, 8)
        else:
            raise ValueError("Invalid receptive field, must be one of [4, 8, 16]")

        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channels, 32, 8, stride=self.strides[0])),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=self.strides[1])),
            nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, stride=self.strides[2])),
            nn.ReLU(),
            init_(nn.Conv2d(128, 64, 3, stride=self.strides[3])),
            nn.ReLU(),
            encoders.Flatten(),
            init_(nn.Linear(self.final_conv_size, self.feature_size)))
        self.train()

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out
