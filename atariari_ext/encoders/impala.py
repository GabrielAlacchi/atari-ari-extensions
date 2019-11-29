import atariari.methods.encoders as encoders


class ImpalaCNN(encoders.ImpalaCNN):

	def forward(self, inputs, fmaps=False):
		return super().forward(inputs)
