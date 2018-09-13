import numpy as np
import time

import ann_methods
import util


def run(searcher, Xq, gt, r):
    """
    Given a searcher, run the search. Return the runtime and the accuracy

    Args:
        searcher (object): Searcher defined in ann_methods.py
        Xq (np.array): Query vectors. shape=(Nq, D). dtype=np.float32
        gt (np.array): Groundtruth. shape=(Nq, ANY). dtype=np.int32
        r (int): Top R

    Returns:
        (float, float): Duration [sec/query] and recall@r over the queries

    """
    assert Xq.ndim == 2
    assert Xq.dtype == np.float32
    Nq = Xq.shape[0]
    I = np.zeros((Nq, r), dtype=int)
    t0 = time.time()
    for i, q in enumerate(Xq):
        I[i] = searcher.search(q=q, topk=r)
    t1 = time.time()
    duration = (t1 - t0) / Nq  # sec/query
    recall = util.recall_at_r(I, gt, r)
    return duration, recall


# Read files
Xt = util.fvecs_read("./data/sift/sift_learn.fvecs")
Xb = util.fvecs_read("./data/sift/sift_base.fvecs")
Xq = util.fvecs_read("./data/sift/sift_query.fvecs")
gt = util.ivecs_read("./data/sift/sift_groundtruth.ivecs")

# Run search
for method in ["rii", "faiss", "falconn", "annoy", "nmslib"]:
    print("=== mehtod: {} ===".format(method))
    searcher = {"annoy": ann_methods.AnnoySearcher,
                "falconn": ann_methods.FalconnSearcher,
                "nmslib": ann_methods.NmslibSearcher,
                "faiss": ann_methods.FaissSearcher,
                "rii": ann_methods.RiiSearcher}[method]()

    print("Start to train:")
    t0 = time.time()
    searcher.train(Xt)
    print("Finish: {} [sec]".format(time.time() - t0))

    print("Start to add:")
    t0 = time.time()
    searcher.add(Xb)
    print("Finish: {} [sec]".format(time.time() - t0))

    r = 1  # recall@1
    duration, recall = run(searcher=searcher, Xq=Xq, gt=gt, r=r)
    print("Runtime/query: {} [msec], Recall@{}: {}".format(duration * 1000, r, recall))
