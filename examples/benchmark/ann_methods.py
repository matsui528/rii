# ANN methods with the shared interface
# The design is inspired by ann-benchmark https://github.com/erikbern/ann-benchmarks

import numpy as np
import annoy  # pip install annoy
import falconn  # pip install FALCONN
import nanopq  # pip install nanopq (can be installed when rii is installed)
import nmslib  # pip install nmslib
import faiss  # conda install -c pytorch faiss-cpu

### If you'd like to debug, please uninstall rii and uncomment the following lines
#import sys
#sys.path.append('../../')

import rii  # pip install rii


class RiiSearcher(object):
    def __init__(self, L=5000, K=1000, M=64):
        self.L = L  # Set None if you'd like to try with other dataset
        self.K = K  # Set None if you'd like to try with other dataset
        self.M = M
        self.index = None

    def train(self, vecs):
        codec = nanopq.PQ(M=self.M, verbose=False).fit(vecs=vecs)
        self.index = rii.Rii(fine_quantizer=codec)

    def add(self, vecs):
        self.index.add_configure(vecs=vecs, nlist=self.K)

    def search(self, q, topk):
        ids, _ = self.index.query(q=q, L=self.L, topk=topk)
        return ids


class AnnoySearcher(object):
    def __init__(self, n_trees=2000, k_search=400):
        self.n_trees = n_trees
        self.k_search = k_search
        self.index = None

    def train(self, vecs):
        pass

    def add(self, vecs):
        self.index = annoy.AnnoyIndex(f=vecs.shape[1], metric="euclidean")
        for n, v in enumerate(vecs):
            self.index.add_item(n, v.tolist())
        self.index.build(self.n_trees)

    def search(self, q, topk):
        return self.index.get_nns_by_vector(q.tolist(), n=topk, search_k=self.k_search)


class FalconnSearcher(object):
    def __init__(self, num_probes=16):
        self.num_probes = num_probes
        self.center = None
        self.params_cp = None
        self.table = None
        self.query_object = None

    def train(self, vecs):
        pass

    def add(self, vecs):
        self.center = np.mean(vecs, axis=0)  # Subtract mean vector later
        self.params_cp = falconn.get_default_parameters(num_points=vecs.shape[0],
                                                        dimension=vecs.shape[1],
                                                        distance=falconn.DistanceFunction.EuclideanSquared,
                                                        is_sufficiently_dense=True)
        # self.params_cp.num_setup_threads = 0  # Single thread mode
        bit = int(np.round(np.log2(vecs.shape[0])))
        falconn.compute_number_of_hash_functions(bit, self.params_cp)

        self.table = falconn.LSHIndex(self.params_cp)
        self.table.setup(vecs - self.center)
        self.query_object = self.table.construct_query_object()

    def search(self, q, topk):
        self.query_object.set_num_probes(self.num_probes)
        return self.query_object.find_k_nearest_neighbors(q - self.center, topk)


class NmslibSearcher(object):
    def __init__(self, post=2, efConstruction=400, efSearch=4):
        self.index = nmslib.init(method='hnsw', space='l2')
        self.post = post
        self.efConstruction = efConstruction
        self.efSearch = efSearch

    def train(self, vecs):
        pass

    def add(self, vecs):
        self.index.addDataPointBatch(vecs)
        self.index.createIndex({'post': self.post, 'efConstruction': self.efConstruction}, print_progress=True)

    def search(self, q, topk):
        self.index.setQueryTimeParams({'efSearch': self.efSearch})
        ids, distances = self.index.knnQuery(q, k=topk)
        return ids


class FaissSearcher(object):
    def __init__(self, nlist=1000, M=64, nprobe=4):
        self.nlist = nlist
        self.M = M
        self.nprobe = nprobe
        self.quantizer = None
        self.index = None

    def train(self, vecs):
        D = vecs.shape[1]
        self.quantizer = faiss.IndexFlatL2(D)
        self.index = faiss.IndexIVFPQ(self.quantizer, D, self.nlist, self.M, 8)
        self.index.train(vecs)

    def add(self, vecs):
        self.index.add(vecs)

    def search(self, q, topk):
        self.index.nprobe = self.nprobe
        D, I = self.index.search(q.reshape(1, -1), topk)
        return I

