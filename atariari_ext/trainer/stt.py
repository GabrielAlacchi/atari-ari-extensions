import torch
import torch.nn as nn
import torch.nn.functional as F
import atariari.methods.jsd_stdim as jsd_stdim


class SpatioTemporalTrainerExt(jsd_stdim.SpatioTemporalTrainer):
	def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
		super().__init__(encoder, config, device, wandb)
