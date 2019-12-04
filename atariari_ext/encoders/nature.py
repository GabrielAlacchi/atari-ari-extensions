import atariari.methods.encoders as encoders


class SelectiveReceptiveFieldNatureCNN(encoders.NatureCNN):

    def __init__(self, input_channels, args, receptive_field=16):
        super().__init__(input_channels, args)

        if receptive_field not in [4, 8, 16]:
            raise ValueError("Invalid receptive field, must be one of [4, 8, 16]")

        self.local_layer = 2 * [4, 8, 16].index(receptive_field)

    @property
    def local_layer_depth(self):
        return self.main[self.local_layer].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:self.local_layer + 2](inputs)
        out = self.main[self.local_layer + 2:](f5)

        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out
