import numpy as np
import pathlib
import nanopq
import pickle
import time
import more_itertools
import texmex_python
import util

### If you'd like to debug, please uninstall rii and uncomment the following lines
#import sys
#sys.path.append('../../')

import rii


def run(engine, L, Xq, gt, r):
    """
    Given a searcher, run the search. Return the runtime and the accuracy

    Args:
        engine (rii.Rii): Rii search engine
        L (int): The number of candidates for search
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
        I[i], _ = engine.query(q=q, topk=r, L=L)
    t1 = time.time()
    duration = (t1 - t0) / Nq  # sec/query
    recall = util.recall_at_r(I, gt, r)
    return duration, recall


# Setup paths
p = pathlib.Path('.')
path_train = p / "data/bigann_learn.bvecs"
path_base = p / "data/bigann_base.bvecs"
path_query = p / "data/bigann_query.bvecs"
path_gt = p / "data/gnd/idx_1000M.ivecs"


# Read queries and groundtruth
Xq = texmex_python.reader.read_bvec(path_query.open("rb")).astype(np.float32)
gt = util.ivecs_read(str(path_gt))


# Reat top Nt vectors for training
print("Start to read training vectors")
Xt = []
Nt = 10000000  # Use top 10M vectors for training
with path_train.open("rb") as f:
    for vec in texmex_python.reader.read_bvec_iter(f):
        Xt.append(vec)
        if len(Xt) == Nt:
            break
Xt = np.array(Xt, dtype=np.float32)
print("Xt.shape: {}, Xt.dtype: {}".format(Xt.shape, Xt.dtype))


# Train a PQ codec and save it
M = 8  # The number of subspace.
path_codec = p / 'cache/codec_m{}.pkl'.format(M)
if not path_codec.exists():
    print("Start to train a codec")
    codec = nanopq.PQ(M=M, verbose=True).fit(vecs=Xt)
    pickle.dump(codec, path_codec.open("wb"))
    print("Dump the codec in {}".format(path_codec))
else:
    print("Read a codec from cache: {}".format(path_codec))
    codec = pickle.load(path_codec.open("rb"))


# Construct a search engine
path_engine = p / 'cache/engine_m{}.pkl'.format(M)
if not path_engine.exists():
    print("Start to construct a Rii engine")
    e = rii.Rii(fine_quantizer=codec)
    batch_size = 10000000
    with path_base.open("rb") as f:
        for n, batch in enumerate(more_itertools.chunked(texmex_python.reader.read_bvec_iter(f), batch_size)):
            print("batch: {} / {}".format(n, int(1000000000 / batch_size)))
            e.add(vecs=np.array(batch, dtype=np.float32))
        e.reconfigure()
    pickle.dump(e, path_engine.open("wb"))
    print("Dump the engine in {}".format(path_engine))
else:
    print("Read an engine from cache: {}".format(path_engine))
    e = pickle.load(path_engine.open("rb"))
e.print_params()


# Run search
r = 1  # Reacll@r
w = 1  # The parameter for search candidates. L = L0 * w = N / nlist * w.   The default (fastest) setting is w=1
duration, recall = run(engine=e, L=e.L0 * w, Xq=Xq, gt=gt, r=r)
print("{} msec/query. Recall@{} = {}".format(duration * 1000, r, recall))

