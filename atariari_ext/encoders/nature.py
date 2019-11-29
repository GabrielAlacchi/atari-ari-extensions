import atariari.methods.encoders as encoders


class NatureCNN(encoders.NatureCNN):

	@property
	def convolution_depth(self):
		return self.main[4].out_channels
