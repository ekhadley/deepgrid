import sys, argparse
from deepgrid.colors import *
import deepgrid as dg
import deepq
import spo

p = argparse.ArgumentParser()

p.add_argument("algo", default="deepq", help="the RL algorithm to use", choices=["spo", "deepq"], nargs="?")
p.add_argument("mode", default="play", help="the mode to run in", choices=["play", "train"], nargs="?")
p.add_argument("--show", help="display picture of agent as it plays (v slow)",  action="store_true")

args = p.parse_args()
print(cyan, args, endc)
print(green, args.mode, endc)

if __name__ == "__main__":
    if args.algo == "deepq":
        if args.mode == "play":
            main = deepq.deepq_play.play
        elif args.mode == "train":
            main = deepq.deepq_train.train
    elif args.algo == "spo":
        if args.mode == "play":
            main = spo.spo_play.play
        elif args.mode == "train":
            main = spo.spo_train.train
    main(show=args.show)
