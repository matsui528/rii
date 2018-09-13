import numpy as np

# These functions are from https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py

def ivecs_read(fname):
    """
    Read a .ivecs file, that contains a set of int-vectors

    Args:
        fname (str): The path to the file

    Returns:
        np.array: shape=(N, D), dtype=np.int32

    """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    """
    Read .fvecs file, that contains a set of float-vectors

    Args:
        fname (str): The path to the file

    Returns:
        np.array: shape=(N, D), dtype=np.float

    """
    return ivecs_read(fname).view('float32')


def recall_at_r(I, gt, r):
    """
    Given a search result and groundtruth, compute Recall@R.
    Note that Recall@R is an averaged value over queries.

    Args:
        I (np.array): Search result for all queries, composed of IDs of the database items.
            dtype=int, shape=(Nq, topk), where Nq is the number of queries.
            I[nq, k] shows the ID of k-th search result for nq-th query.
        gt (np.array): Groundtruth IDs for all queries. This is typically from groundtruth.ivecs for BIGANN data.
            dtype=int32, shape=(Nq, ANY). We only use gt[:, 0]. where gt[nq] is the ID of the nearest item from
            the database for nq-the query.
        r (int): The R of Recall@R.

    Returns:
        float: The average Recall@R over the queries

    """
    assert I.ndim == 2
    assert gt.ndim == 2
    Nq, topk = I.shape
    assert r <= topk
    n_ok = (I[:, :r] == gt[:, :1]).sum()
    return n_ok / float(Nq)