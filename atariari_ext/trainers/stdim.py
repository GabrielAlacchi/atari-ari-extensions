import atariari.methods.stdim as stdim
import torch
import torch.nn as nn


class InfoNCESpatioTemporalTrainer(stdim.InfoNCESpatioTemporalTrainer):

	def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
		super().__init__(encoder, config, device=device, wandb=wandb)
		self.classifier1 = nn.Linear(
			self.encoder.hidden_size, 
			self.encoder.convolution_depth).to(device)
		self.classifier2 = nn.Linear(
			self.encoder.convolution_depth, 
			self.encoder.convolution_depth).to(device)
