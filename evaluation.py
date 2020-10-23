import numpy as np
from utils import flatten

def cosine_sim(A, B):
    img_norm = np.linalg.norm(A, axis=1)
    caps_norm = np.linalg.norm(B, axis=1)
    scores = np.dot(A, B.T)
    norms = np.dot(np.expand_dims(img_norm, 1),
                   np.expand_dims(caps_norm.T, 1).T)
    scores = (scores / norms)
    return scores

def recall_at_k_multi_cap(imgs_enc, caps_enc, ks=[1, 5, 10], scores=None):
    if scores is None:
        scores = cosine_sim(imgs_enc[::5, :], caps_enc)

    ranks = np.array([np.nonzero(np.in1d(row, np.arange(x * 5, x * 5 + 5, 1)))[0][0]
                      for x, row in enumerate(np.argsort(scores, axis=1)[:, ::-1])])

    medr_caps_search = np.median(ranks)

    recall_caps_search = list()

    for k in [1, 5, 10]:
        recall_caps_search.append(
            (float(len(np.where(ranks < k)[0])) / ranks.shape[0]) * 100)

    ranks = np.array([np.nonzero(row == int(x / 5.0))[0][0]
                      for x, row in enumerate(np.argsort(scores.T, axis=1)[:, ::-1])])

    medr_imgs_search = np.median(ranks)

    recall_imgs_search = list()
    for k in ks:
        recall_imgs_search.append(
            (float(len(np.where(ranks < k)[0])) / ranks.shape[0]) * 100)

    return recall_caps_search, recall_imgs_search, medr_caps_search, medr_imgs_search


def avg_recall(imgs_enc, caps_enc):
    """ Compute 5 fold recall on set of 1000 images """
    res = list()
    if len(imgs_enc) % 5000 == 0:
        max_iter = len(imgs_enc)
    else:
        max_iter = len(imgs_enc) - 5000

    for i in range(0, max_iter, 5000):
        imgs = imgs_enc[i:i + 5000]
        caps = caps_enc[i:i + 5000]
        res.append(recall_at_k_multi_cap(imgs, caps))

    return [np.sum([x[i] for x in res], axis=0) / len(res) for i in range(len(res[0]))]


def eval_recall(imgs_enc, caps_enc):

    imgs_enc = np.vstack(flatten(imgs_enc))
    caps_enc = np.vstack(flatten(caps_enc))
    res = avg_recall(imgs_enc, caps_enc)
    return res
