import atariari.methods.encoders as encoders
import torch.nn.functional as F


class ImpalaCNN(encoders.ImpalaCNN):
 
    @property
    def convolution_depth(self):
        return self.depths[-2]

    def forward(self, inputs, fmaps=False):
        f5 = self.layer3(self.layer2(self.layer1(inputs)))

        if not self.downsample:
            out = self.layer4(f5)
        else:
            out = f5

        out = F.relu(self.final_linear(self.flatten(f5)))

        if fmaps:
            return {
                "f5": f5.permute(0, 2, 3, 1),
                "out": out
            }

        return out
