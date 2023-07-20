import cv2, numpy as np
from tqdm import trange
import math

purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
green = '\033[92m'
yellow = '\033[93m'
red = '\033[91m'
endc = '\033[0m'
bold = '\033[1m'
underline = '\033[4m'

def sampleDist(probs, returnProb=False):
    summ = sum(probs)
    assert round(summ, 3) <= 1, f"distribution probabilities should sum to ~1. sum is {summ}"
    r = np.random.uniform(0, 1)
    cum = [sum(probs[0:i+1]) for i in range(len(probs))]
    for i, c in enumerate(cum):
        if r < c:
            if returnProb: return i, probs[i]
            return i

def imscale(img, s):
    try:
        w, h, d = np.shape(img)
    except:
        w, h = np.shape(img)
    assert not 0 in [w, h], "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)