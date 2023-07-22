import sys, argparse, os
from deepgrid.colors import *
import deepgrid as dg
import deepq
import vanilla

p = argparse.ArgumentParser()

p.add_argument("algo", default="spo", help="the RL algorithm to use", choices=["spo", "deepq"], nargs="?")
p.add_argument("mode", default="play", help="the mode to run in", choices=["play", "train"], nargs="?")
p.add_argument("--show", help="display picture of agent as it plays (v slow)",  action="store_true")
p.add_argument("--profile", help="profile the code, save in same dir as this",  action="store_true")

args = p.parse_args()
print(cyan, args, endc)

if args.profile:
    from cProfile import Profile
    prof = Profile()
    prof.enable()

if __name__ == "__main__":
    if args.algo == "deepq":
        if args.mode == "play":
            main = deepq.deepq_play.play
        elif args.mode == "train":
            main = deepq.deepq_train.train
    elif args.algo == "spo":
        if args.mode == "play":
            main = vanilla.vpg_play.play
        elif args.mode == "train":
            main = vanilla.rtg_train.train
    main(show=args.show)

if args.profile:
    pth = os.path.dirname(os.path.abspath(__file__))
    prof.dump_stats(f"{pth}\\tmp")