import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
lime = '\033[92m'
yellow = '\033[93m'
red = "\033[38;5;196m"
pink = "\033[38;5;206m"
orange = "\033[38;5;202m"
green = "\033[38;5;34m"
gray = "\033[38;5;8m"

bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def sampleDist(probs, returnProb=False):
    summ = sum(probs)
    assert round(summ, 3) <= 1, f"distribution probabilities should sum to ~1. sum is {summ}"
    r = np.random.uniform(0, 1)
    cum = [sum(probs[0:i+1]) for i in range(len(probs))]
    for i, c in enumerate(cum):
        if r < c:
            if returnProb: return i, probs[i]
            return i

def entropy(probs):
    if isinstance(probs, np.ndarray): return -sum([p*np.log(p) for p in probs])
    if isinstance(probs, torch.Tensor): return -torch.sum(probs*torch.log(probs), axis=1)

def imscale(img, s):
    try:
        w, h, d = np.shape(img)
    except:
        w, h = np.shape(img)
    assert w*h > 0, "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)