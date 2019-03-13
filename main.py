from argparse import ArgumentParser
from src.train_trial import train
from src.train_trial import trial
import sys


def main():
    parser = ArgumentParser("OpenAiGym_algorithm")
    parser.add_argument('algorithm', type=str,
                        help='algorithm specified in src/algorithm/')
    subparsers = parser.add_subparsers(help='mode')

    parser_rr = subparsers.add_parser('rr')
    parser_rr_subparsers = parser_rr.add_subparsers(help='choose between train or trial')

    parser_rr_train = parser_rr_subparsers.add_parser('train', help='train mode in real environment')
    parser_rr_train.add_argument('hyperparameters', type=str,
                                 help='.csv folder with hyperparameters for specified algorithm')
    parser_rr_train.add_argument('outdir', type=str,
                                 help='output directory of your training data')
    parser_rr_trial = parser_rr_subparsers.add_parser('trial', help='trial mode in real environment')
    parser_rr_trial.add_argument('policy', type=str,
                                 help='path to your policy')
    parser_rr_trial.add_argument('outdir', type=str,
                                 help='save your results in specified directory')
    parser_rr_trial.add_argument('episodes', type=int,
                                 help='number of episodes to start your trial in rr mode')

    # parser_rr_trial specification

    parser_sim = subparsers.add_parser('sim')
    parser_sim_subparsers = parser_sim.add_subparsers(help='choose between train or trial')

    parser_sim_train = parser_sim_subparsers.add_parser('train', help='train mode in simulated environment')
    parser_sim_train.add_argument('hyperparameters', type=str,
                                 help='.csv folder with hyperparameters for specified algorithm')
    parser_sim_train.add_argument('outdir', type=str,
                                 help='output directory of your training data')

    parser_sim_trial = parser_sim_subparsers.add_parser('trial', help='trial mode in simulated environment')
    parser_sim_trial.add_argument('policy', type=str,
                                 help='path to your policy')
    parser_sim_trial.add_argument('outdir', type=str,
                                 help='save your results in specified directory')
    parser_sim_trial.add_argument('episodes', type=int,
                                 help='number of episodes to start your trial in sim mode')

    args = parser.parse_args()

    print(sys.argv[2])


    if(sys.argv[3] == 'train'):
        train(run_configs=args.hyperparameters, algorithm=args.algorithm,
              outdir=args.outdir, mode=sys.argv[2])
        return
    if(sys.argv[3] =='trial'):
        trial(algorithm=args.algorithm, policy=args.policy, mode=sys.argv[2],
             episodes=args.episodes, outdir=args.outdir)
        return

    raise Exception("There exist only train and trial modes")


main()
