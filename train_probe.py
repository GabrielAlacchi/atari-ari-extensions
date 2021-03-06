import wandb
import sys
import torch
import pickle

from atariari.methods.utils import get_argparser
from atariari.benchmark.episodes import get_episodes
from atariari_ext.benchmark.probe import WandbLoggingProbeTrainer
from atariari.methods.encoders import ImpalaCNN
from atariari_ext.encoders.nature import AdjustableNatureCNN


parser = get_argparser()
parser.add_argument('--entity', default='neurips-challenge', dest='entity')
parser.add_argument('--project', default='atari-ari', dest='project')
parser.add_argument('--run-id', type=str, required=True, dest='run_id')
parser.add_argument('--run-name', type=str, default='', dest='run_name')
parser.add_argument('--receptive-field', default=16, type=int, dest='receptive_field')
parser.add_argument('--frame-file', default='', type=str, dest='frame_file')
args = parser.parse_args(sys.argv[1:])

encoder_model_file = wandb.restore('encoder.pt', run_path=f'{args.entity}/{args.project}/{args.run_id}', replace=True)

device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

state_dict = torch.load(encoder_model_file.name, map_location=device)

if args.frame_file:
    with open(args.frame_file, 'rb') as f:   
        tr_episodes, val_episodes,\
        tr_labels, val_labels,\
        test_episodes, test_labels = pickle.load(f)
else:
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

observation_shape = tr_episodes[0][0].shape

if args.encoder_type == 'Impala':
    encoder = ImpalaCNN(observation_shape[0], args)
else:
    encoder = AdjustableNatureCNN(observation_shape[0], args)

encoder.to(device)
encoder.load_state_dict(state_dict)

run_name = args.run_name if args.run_name else f'probe-{args.encoder_type}-{args.env_name}'.lower()

wandb.init(name=run_name,
           entity=args.entity,
           project=args.project)

config = {}
config.update(vars(args))
config['obs_space'] = observation_shape  # weird hack

wandb.config.update(config)

probe_trainer = WandbLoggingProbeTrainer(encoder, representation_len=args.feature_size, wandb=wandb)
probe_trainer.train(tr_episodes, val_episodes, tr_labels, val_labels)
probe_trainer.test(test_episodes, test_labels)
