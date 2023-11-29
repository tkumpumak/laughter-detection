import numpy as np


def findnextgap(k, convprobs):
    while k < len(convprobs):
        if convprobs[k] == 0:
            return k
        k += 1
    return k


def findedge(k, convprobs, probs, cutoff_threshold):
    while k < len(convprobs):
        if probs[k] < cutoff_threshold:
            convprobs[k] = 0
        else:
            return findnextgap(k, convprobs)
        k += 1
    return k


def condition_propabilities(probs, threshold, cutoff_threshold, fps):
    probs = probs.copy()

    convprobs = probs > threshold
    span = 2
    sec_kern = np.full(int(fps * span), True)
    convprobs = np.convolve(1 * convprobs, 1 * sec_kern, "same")
    convprobs[convprobs > 1] = 1

    k = 0
    while k < len(convprobs):
        if convprobs[k] == 1:
            k = findedge(k, convprobs, probs, cutoff_threshold)
        k += 1

    convprobs = np.flipud(convprobs)
    probs = np.flipud(probs)

    # same backwards
    k = 0
    while k < len(convprobs):
        if convprobs[k] == 1:
            k = findedge(k, convprobs, probs, cutoff_threshold)
        k += 1
    convprobs = np.flipud(convprobs)

    return convprobs
