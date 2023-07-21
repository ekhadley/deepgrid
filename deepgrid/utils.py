import cv2, numpy as np

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