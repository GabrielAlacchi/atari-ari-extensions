from atariari.benchmark.episodes import get_episodes
from atariari.methods.utils import get_argparser
import pickle
import sys


parser = get_argparser()
args = parser.parse_args(sys.argv[1:])

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

with open('%s.pickle' % args.env_name, 'wb') as f:
	pickle.dump((
		tr_episodes, val_episodes,\
		tr_labels, val_labels,\
		test_episodes, test_labels
	), f)
