import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from atariari.methods.dim_baseline import DIMTrainer
from atariari.methods.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from atariari.methods.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from atariari.methods.jsd_stdim import SpatioTemporalTrainer
from atariari.methods.utils import get_argparser
from atariari.methods.encoders import NatureCNN, ImpalaCNN
from atariari.methods.cpc import CPCTrainer
from atariari.methods.vae import VAETrainer
from atariari.methods.no_action_feedforward_predictor import NaFFPredictorTrainer
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
import wandb
from atariari.benchmark.episodes import get_episodes

from atariari_ext.trainer.stt import InfoNCESpatioTemporalTrainerExt

def train_encoder(args, p=None):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    tr_eps, val_eps = get_episodes(steps=args.pretraining_steps,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=1,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.probe_collect_mode,
                                 train_mode="train_encoder",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)

    observation_shape = tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape
    if p:
      trainer = InfoNCESpatioTemporalTrainerExt(encoder, config, device=device, wandb=wandb, p=p)
    else:
      trainer = InfoNCESpatioTemporalTrainerExt(encoder, config, device=device, wandb=wandb)
    
    trainer.train(tr_eps, val_eps)

    return encoder


ps = [0.1, 0.3, 0.5, 0.7, 0.9]
envs = ['SpaceInvadersNoFrameskip-v4']

for env in envs:
    parser = get_argparser()
    args = parser.parse_args([
      '--env-name',
      env,'--entropy-threshold','0.6']
      )

    for p in ps:
        wandb.init(name='weighted-loss-p-%.2f-%s' % (p, args.env_name.lower()),
               project='atari-ari')

        encoder = train_encoder(args, p)
