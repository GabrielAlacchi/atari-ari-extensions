import wandb
import sys
import torch

from atariari.methods.utils import get_argparser
from atariari.benchmark.episodes import get_episodes
from atariari_ext.benchmark.probe import WandbLoggingProbeTrainer


parser = get_argparser()
parser.add_argument('--entity', default='neurips-challenge', dest='entity')
parser.add_argument('--project', default='atari-ari', dest='project')
parser.add_argument('--run-id', type=str, required=True, dest='run_id')
args = parser.parse_args(sys.argv[1:])

encoder_model_file = wandb.restore('encoder.pt', run_path=f'{args.entity}/{args.project}/{args.run_id}')

device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

encoder = torch.load(encoder_model_file.name, map_location=device)

tr_episodes, val_episodes,\
tr_labels, val_labels,\
test_episodes, test_labels = get_episodes(
    steps=args.pretraining_steps,
    env_name=args.env_name,
    seed=args.seed,
    num_processes=1,
    num_frame_stack=args.num_frame_stack,
    downsample=not args.no_downsample,
    color=args.color,
    entropy_threshold=args.entropy_threshold,
    collect_mode=args.probe_collect_mode,
    checkpoint_index=args.checkpoint_index,
    min_episode_length=args.batch_size)

wandb.init(name=f'probe-{args.encoder_type}-{args.env_name}', entity=args.entity, project=args.project)

observation_shape = tr_episodes[0][0].shape

config = {}
config.update(vars(args))
config['obs_space'] = observation_shape  # weird hack

wandb.config.update(config)

trainer = WandbLoggingProbeTrainer(encoder, representation_len=encoder.feature_size, wandb=wandb)
probe_trainer.train(tr_episodes, val_episodes,
                     tr_labels, val_labels,)
probe_trainer.test(test_episodes, test_labels)
