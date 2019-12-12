import torch
import torch.nn as nn
import torch.nn.functional as F
import atariari.methods.stdim as stdim


class InfoNCESpatioTemporalTrainerExt(stdim.InfoNCESpatioTemporalTrainer):
	def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None, p=0.5):
		self.p = p
		super().__init__(encoder, config, device, wandb)


	def do_one_epoch(self, epoch, episodes):
		mode = "train" if self.encoder.training and self.classifier1.training else "val"
		epoch_loss, accuracy, steps = 0., 0., 0
		accuracy1, accuracy2 = 0., 0.
		epoch_loss1, epoch_loss2 = 0., 0.
		data_generator = self.generate_batch(episodes)
		for x_t, x_tprev in data_generator:
			f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tprev, fmaps=True)

			# Loss 1: Global at time t, f5 patches at time t-1
			f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
			sy = f_t_prev.size(1)
			sx = f_t_prev.size(2)

			N = f_t.size(0)
			loss1 = 0.
			for y in range(sy):
				for x in range(sx):
					predictions = self.classifier1(f_t)
					positive = f_t_prev[:, y, x, :]
					logits = torch.matmul(predictions, positive.t())
					step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
					loss1 += step_loss
			loss1 = loss1 / (sx * sy)

			# Loss 2: f5 patches at time t, with f5 patches at time t-1
			f_t = f_t_maps['f5']
			loss2 = 0.
			for y in range(sy):
				for x in range(sx):
					predictions = self.classifier2(f_t[:, y, x, :])
					positive = f_t_prev[:, y, x, :]
					logits = torch.matmul(predictions, positive.t())
					step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
					loss2 += step_loss
			loss2 = loss2 / (sx * sy)

			loss = self.p * loss1 + (1-self.p) * loss2

			if mode == "train":
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			epoch_loss += loss.detach().item()
			epoch_loss1 += loss1.detach().item()
			epoch_loss2 += loss2.detach().item()
			steps += 1
		self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps, prefix=mode)
		if mode == "val":
			self.early_stopper(-epoch_loss / steps, self.encoder)
