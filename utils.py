import cv2, numpy as np
from tinygrad.nn import Tensor

purple = '\033[95m'
blue = '\033[94m'
cyan = '\033[96m'
green = '\033[92m'
yellow = '\033[93m'
red = '\033[91m'
endc = '\033[0m'
bold = '\033[1m'
underline = '\033[4m'

def imscale(img, s):
    try:
        w, h, d = np.shape(img)
    except:
        w, h = np.shape(img)
    assert not 0 in [w, h], "empty src image"
    return cv2.resize(img, (round(len(img[0])*s), round(len(img)*s)), interpolation=cv2.INTER_NEAREST)

def isint(val):
    return isinstance(val, (int, np.integer))

def sampleDist(probs):
    assert sum(probs) <= 1, f"distribution probabilities should sum to 1"
    r = np.random.uniform(0, 1)
    cum = [sum(probs[0:i+1]) for i in range(len(probs))]
    for i, c in enumerate(cum):
        if r < c: return i