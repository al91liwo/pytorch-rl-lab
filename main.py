from argparse import ArgumentParser
from train import train

def real(algorithm):
    print(algorithm)
def test(algorithm):
    print(algorithm)
def main():
    parser = ArgumentParser("OpenAiGym algorithm")
    parser.add_argument('algorithm', type=str,
                        help='algorithm specified in src/algorithm/')
    parser.add_argument('mode', type=str,
                        help='one of: "test"\n'
                             '"train"\n'
                             '"real"')
    parser.add_argument('hyperparameters', type=str,
                        help='.csv file containing rows for each hyperparameter-set to test')
    parser.add_argument('outdir', type=str, help='Directory that contains the results of the runs')

    args = parser.parse_args()

    mode = args.mode.lower()
    algorithm = args.algorithm
    hyperparameters = args.hyperparameters
    outdir = args.outdir

    print(mode)
    if mode == "train":
        train(run_configs=hyperparameters, algorithm=algorithm, outdir=outdir)
        return
    if mode == "real":
        real(algorithm=algorithm)
        return
    if mode == "test":
        test(algorithm=algorithm)
        return

    # default
    if mode == "":
        print('no mode selected, choosing "train" as default')
        train(run_configs=hyperparameters, algorithm=algorithm, outdir=outdir)
        return

    raise Exception('no valid mode selected, run: "main.py --help"')

main()
