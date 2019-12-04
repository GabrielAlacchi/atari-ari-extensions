import time
import atariari_ext
import wandb

from collections import deque
from itertools import chain

import numpy as np
import sys
import torch

from atariari.methods.dim_baseline import DIMTrainer
from atariari.methods.utils import get_argparser
from atariari.methods.encoders.nature import NatureCNN, ImpalaCNN
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
from atariari.benchmark.episodes import get_episodes
from atariari_ext.encoders.nature import SelectiveReceptiveFieldNatureCNN


parser = get_argparser()
parser.add_argument('--entity', default='neurips-challenge', dest='entity')
parser.add_argument('--project', default='atari-ari', dest='project')
parser.add_argument('--receptive-field', default=16, type=int, dest='receptive_field')
args = parser.parse_args(sys.argv[1:])

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
elif args.encoder_type == "ReceptiveNature":
    encoder = SelectiveReceptiveFieldNatureCNN(observation_shape[0], args, receptive_field=args.receptive_field)
elif args.encoder_type == "Impala":
    encoder = ImpalaCNN(observation_shape[0], args)

encoder.to(device)
torch.set_num_threads(1)

wandb.init(name='%s-cnn-%s' % (args.encoder_type, args.env_name),
           entity=args.entity,
           project=args.project)

config = {}
config.update(vars(args))
config['obs_space'] = observation_shape  # weird hack

wandb.config.update(config)

trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)

trainer.train(tr_eps, val_eps)
