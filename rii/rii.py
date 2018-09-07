import main
import nanopq
import numpy as np
import copy

class Rii(object):
    """Reconfigurable Inverted Index (Rii) [Matsui18]_.

    Fast and memory efficient approximate nearest neighbor search method based on IVFADC.
    In addition to the usual ANN search, Rii provides two useful functionalities:
    (1) efficient search over a subset of the whole dataset,
    and (2) reconfiguration of the data structure such that the search remains fast even after
    many items are newly added.

    .. [Matsui18] Y. Matsui, R. Hinami, and S. Satoh, "Reconfigurable Inverted Index", ACM Multimedia 2018

    Args:
        fine_quantizer (object): The instance for encoding/decoding vectors.
            `nanopq.PQ <https://nanopq.readthedocs.io/en/latest/source/api.html#product-quantization-pq>`_ or
            `nanopq.OPQ <https://nanopq.readthedocs.io/en/latest/source/api.html#optimized-product-quantization-opq>`_.
            This must have been already trained.

    Attributes:
        fine_quantizer (object): The instance for encoding/decoding vectors.
            `nanopq.PQ <https://nanopq.readthedocs.io/en/latest/source/api.html#product-quantization-pq>`_ or
            `nanopq.OPQ <https://nanopq.readthedocs.io/en/latest/source/api.html#optimized-product-quantization-opq>`_.
        threshold (object): The threshold function for the subset-search.
            An instance of `np.poly1d <https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html>`_ class.
            Given ``L``, compute the threshold for the selection of the search method: ``S_thre = threshold(L)``.

    """
    def __init__(self, fine_quantizer):
        assert isinstance(fine_quantizer, nanopq.PQ) or isinstance(fine_quantizer, nanopq.OPQ)
        assert fine_quantizer.codewords is not None, "Please fit the PQ/OPQ instance first"
        assert fine_quantizer.Ks <= 256, "Ks must be less than 256 so that each code must be uint8"
        self.fine_quantizer = copy.deepcopy(fine_quantizer)  # PQ/OPQ instance
        self.impl_cpp = main.RiiCpp(fine_quantizer.codewords, fine_quantizer.verbose)
        self.threshold = None  # threshold function for subset search

    @property
    def M(self):
        """int: The number of sub-spaces for PQ/OPQ"""
        return self.fine_quantizer.M

    @property
    def Ks(self):
        """int: The number of codewords for each sub-space, typically 256."""
        return self.fine_quantizer.Ks

    @property
    def N(self):
        """int: The number of vectors (PQ-codes)"""
        return self.impl_cpp.N

    @property
    def nlist(self):
        """int: The number of posting lists"""
        return self.impl_cpp.nlist

    @property
    def codewords(self):
        """np.ndarray: The codewords for PQ/OPQ, where
        shape=(M, Ks, D/M) with dtype=np.float32. Note that
        ``codewords[m][ks]`` specifies ks-th codeword (Ds-dim) for m-th subspace.
        """
        return self.fine_quantizer.codewords

    @property
    def coarse_centers(self):
        """np.ndarray: The centers for coarse assignments, where as they are also PQ-codes.
        Note that shape=(nlist, M) with dtype=np.uint8. If coarse_centers have not been created yet
        (i.e., :func:`reconfigure` or :func:`add_configure` has not been called), this returns None.
        """
        if self.nlist == 0:
            return None
        else:
            return np.array(self.impl_cpp.coarse_centers,
                            dtype=self.fine_quantizer.code_dtype)

    @property
    def codes(self):
        """np.ndarray: The database PQ-codes, where shape=(N, M) with dtype=np.uint8.
        Accessing this property would be slow because the whole data
        is converted online from std::vector<unsigned char> in the cpp-instance to np.array.
        If vectors have not been added yet (i.e., :func:`add` or :func:`add_configure` has not been called),
        this returns None.
        """
        if self.N == 0:
            return None
        else:
            return np.array(self.impl_cpp.flattened_codes,
                            dtype=self.fine_quantizer.code_dtype).reshape(self.N, self.M)

    @property
    def posting_lists(self):
        """list: The posting lists for assignment of
        identifiers. List of list. Note that ``len(posting_lists) == nlist``,
        and ``posting_lists[no]`` contains the IDs of PQ-codes
        (a list of int), where their nearest coarse_center is no-th one.
        """
        return self.impl_cpp.posting_lists

    @property
    def verbose(self):
        """bool: Verbose flag. Rewritable."""
        return self.impl_cpp.verbose

    @verbose.setter
    def verbose(self, v):
        self.fine_quantizer.verbose = v
        self.impl_cpp.verbose = v

    @property
    def L0(self):
        """int: The average length of each posting list,
        i.e., ``L0 == int(round(N/nlist))``. If nlist=0, return None.
        """
        if self.nlist == 0:
            return None
        else:
            return int(np.round(self.N / self.nlist))

    def reconfigure(self, nlist=None, iter=5):
        """Given a new ``nlist``, update the :attr:`coarse_centers` and
        :attr:`posting_lists` by grouping the stored PQ-codes (:attr:`codes`)
        into ``nlist`` clusters.

        You can call this if you find
        the search becomes slower after you add too many new items.
        After the reconfiguration, the :attr:`threshold` is also updated.
        See Alg. 3 in the paper for more details.

        Args:
            nlist (int): The new number of posting lists. If None is specified
                (dafault value), nlist=sqrt(N) is automatically set.
            iter (int): The number of iteration for pqk-means to update
                :attr:`coarse_centers`.

        """
        if nlist is None:
            # As a default value, nlist is set as sqrt(N) as suggested by
            # https://github.com/facebookresearch/faiss/wiki/Index-IO,-index-factory,-cloning-and-hyper-parameter-tuning#guidelines
            nlist = int(np.sqrt(self.N))

        assert 0 < nlist

        self.impl_cpp.reconfigure(nlist, iter)

        self.threshold = estimate_best_threshold_function(
            e=self, queries=self.fine_quantizer.decode(self.codes[:min(100, self.N)]))

    def add(self, vecs, update_posting_lists="auto"):
        """Push back new vectors to the system.
        More specifically, this function
        (1) encodes the new vectors into PQ/OPQ-codes,
        (2) pushes back them to the existing database PQ/OPQ-codes, and
        (3) finds the nearest coarse-center for each code and
        updates the corresponding posting list.
        Note that (3) is done only if (a) ``update_posting_lists=True`` or
        (b) ``update_posting_lists='auto'`` and 0 < :attr:`nlist`.

        In usual cases, update_posting_lists should be True.
        It will be False only if
        (1) this is the first call of data addition, i.e., coarse_centers
        have not been created yet. This is the case when :func:`add` is called inside :func:`add_configure`.
        Usually, you don't need to care about this case.
        (2) you plan to call :func:`add` several times (possibly because you read data in a batch way)
        and then run :func:`reconfigure`. In such case, you can skip updating posting lists
        because they will be again updated in :func:`reconfigure`.

        If you set 'auto' for update_posting_lists, it will be automatically set correctly, i.e.,
        True is set if the posting lists have alreay been created (0 < `nlist`), and False is set otherwise.


        Args:
            vecs (np.ndarray): The new vectors with the shape=(Nv, D) and dtype=np.float32,
                where Nv is the number of new vectors.
            update_posting_lists (bool or str): True or False or 'auto'. If True, :attr:`posting_lists` will be updated.
                This should be True for usual cases.

        """
        assert vecs.ndim == 2
        assert vecs.dtype == np.float32

        self.impl_cpp.add_codes(self.fine_quantizer.encode(vecs),
                                self._resolve_update_posting_lists_flag(update_posting_lists))

    def add_configure(self, vecs, nlist=None, iter=5):
        """Run :func:`add` (with ``update_postig_lists=False``) and :func:`reconfigure`.
        To add vectors for the first time, please call this.

        Args:
            vecs (np.ndarray): The new vectors with the shape=(Nv, D) and dtype=np.float32,
                where Nv is the number of new vectors.
            nlist (int): The new number of posting lists. If None is specified
                (dafault value), nlist=sqrt(N) is automatically set.
            iter (int): The number of iteration for pqk-means to update
                :attr:`coarse_centers`.

        Returns:
            self

        """
        self.add(vecs=vecs, update_posting_lists=False)
        self.reconfigure(nlist=nlist, iter=iter)
        return self

    def merge(self, engine, update_posting_lists='auto'):
        """Given another Rii instance (engine), merge its PQ-codes (engine.codes)
        to this instance. IDs for new PQ-codes are automatically assigned.
        For example, if self.N = 100, the IDs of the merged PQ-codes would be 100, 101, ...

        The original posting lists of this instance will be kept
        maintained; new PQ-codes will be simply added over that.
        Thus it might be better to call :func:`reconfigure` after many new codes are merged.

        Args:
            engine (rii.Rii): Rii instance. engine.fine_quantizer should be
                the same as self.fine_quantizer
            update_posting_lists (bool or str): True or False or 'auto'. If True, :attr:`posting_lists` will be updated.
                This should be True for usual cases.

        """
        assert isinstance(engine, Rii)
        assert self.fine_quantizer == engine.fine_quantizer, \
            "Two engines to be merged must have the same fine quantizer"

        if engine.N != 0:
            self.impl_cpp.add_codes(engine.codes,
                                    self._resolve_update_posting_lists_flag(update_posting_lists))

        if self.verbose:
            print("The number of codes: {}".format(self.N))

    def query(self, q, topk=1, L=None, target_ids=None, sort_target_ids=True, method="auto"):
        """Given a query vector, run the approximate nearest neighbor search over the stored PQ-codes.
        This functions returns the identifiers and the distances of ``topk`` nearest PQ-codes to the query.

        The search can be conducted over a subset of the whole PQ-codes by specifying ``target_ids``.
        For example, if ``target_ids=np.array([23, 567, 998])``, the search result would be
        the items with these identifiers, sorted by the distance to the query.

        Inside this function, the algorithm for the search is selected either from PQ-linear-scan (see Alg. 1
        in the paper) or inverted-index (see Alg. 2 in the paper). This is specified by ``method``, by setting
        'linear' or 'ivf'. If 'auto' is set, the faster one is automatically selected
        (See Alg. 3 in the paper for more details).

        See :ref:`guideline_for_search` for tips of the parameter selection.

        Args:
            q (np.ndarray): The query vector with the shape=(D, ) and dtype=np.float32.
            topk (int): The number of PQ-codes to be returned. The default value is 1.
            L (int): The number of PQ-codes for the candidates of distance evaluation.
                With a higher ``L`` value, the accuracy is boosted but the runtime gets slower.
                The default value is a minimum multiple of :attr:`L0` that covers ``topk``.
                This is typically :attr:`L0`.
                Note that ``L`` is used only if the search method is inverted-index.
            target_ids (np.ndarray): The target identifiers with the shape=(S, ) and dtype=np.int64, where S can
                be any scalar. The default value is None, then the search is run over the whole dataset.
                Note that ``target_ids`` does not need to be sorted if ``sort_target_ids==True``, where
                it will be sorted inside this function automatically. Otherwise please sort ``target_ids``
                before calling this function.
            sort_target_ids (bool): The default value is True. If True, ``target_ids`` will be sorted automatically
                inside this function before running the search. If False, ``target_ids`` will not be sorted.
                Note that ``target_ids`` must be sorted before the search, so please sort it by yourself
                if you set ``sort_target_ids`` as False.
            method (str): The search algorithm to be used: 'linear', 'ivf', or 'auto'.

        Returns:
            (np.ndarray, np.ndarray):
                The result (nearest items) of the search.
                The first one is the identifiers of the items, with the shape=(topk, ) and dtype=int64.
                The second one is the distances of the items to the query, with the shape=(topk, ) and dtype=float64

        """
        assert 0 < self.N   # Make sure there are codes to be searched
        assert 0 < self.nlist   # Make sure posting lists are available
        assert method in ["auto", "linear", "ivf"]

        if topk is None:
            topk = self.N
        assert 1 <= topk <= self.N

        if L is None:
            L = self._multiple_of_L0_covering_topk(topk=topk)
        assert topk <= L <= self.N,\
            "Parameters are weird. Make sure topk<=L<=N:  topk={}, L={}, N={}".format(topk, L, self.N)

        if target_ids is None:
            tids = np.array([], dtype=np.int64)
            len_target_ids = self.N
        else:
            assert isinstance(target_ids, np.ndarray)
            assert target_ids.dtype == np.int64
            assert target_ids.ndim == 1
            if sort_target_ids:
                tids = np.sort(target_ids)
            else:
                tids = target_ids
            len_target_ids = len(tids)
        assert topk <= len_target_ids <= self.N, \
            "Parameters are weird. Make sure topk<=len(target_ids)<=N:  "\
            "topk={}, len(target_ids)={}, N={}".format(topk, len_target_ids, self.N)

        if isinstance(self.fine_quantizer, nanopq.OPQ):
            q_ = self.fine_quantizer.rotate(q)
        elif isinstance(self.fine_quantizer, nanopq.PQ):
            q_ = q

        if method == "auto":
            if self._use_linear(len_target_ids, L):
                ids, dists = self.impl_cpp.query_linear(q_, topk, tids)
            else:
                ids, dists = self.impl_cpp.query_ivf(q_, topk, tids, L)
        elif method == "linear":
            ids, dists = self.impl_cpp.query_linear(q_, topk, tids)
        elif method == "ivf":
            ids, dists = self.impl_cpp.query_ivf(q_, topk, tids, L)

        return np.array(ids), np.array(dists)

    def clear(self):
        """Clear all data, i.e., (1) coarse_centers, (2) PQ-codes, (3) posting_lists, and (4) threshold function.
        Note that codewords are kept.

        """
        self.threshold = None
        self.impl_cpp.clear()

    def print_params(self):
        """Print all parameters with some verbose information for debugging.

        """
        print("verbose:", self.verbose)
        print("M:", self.M)
        print("Ks:", self.Ks)
        print("fine_quantizer:", self.fine_quantizer)
        print("N:", self.N)
        print("nlist:", self.nlist)
        print("L0:", self.L0)
        print("cordwords.shape:", self.codewords.shape)
        if self.nlist == 0:
            print("coarse_centers.shape:", None)
        else:
            print("coarse_centers.shape:", self.coarse_centers.shape)

        if self.codes is None:
            print("codes.shape:", None)
        else:
            print("codes.shape:", self.codes.shape)

        print("[len(poslist) for poslist in posting_lists]: [", end="")
        for n, poslist in enumerate(self.posting_lists):
            if 10 < n:
                print(" ...", end="")
                break
            print(str(len(poslist)) + ", ", end="")
        print("]")
        for topk in 1, 10, 100:
            if self.nlist == 0:
                L = "None"
            else:
                L = self._multiple_of_L0_covering_topk(topk)
            print("_multiple_of_L0_covering_topk(topk={}): {}".format(topk, L))
        print("threshold function thre_{|S|}=f(L):", self.threshold)
        for S in [10 ** (2 + n) for n in range(5)]:
            if self.threshold is None:
                use_linear = None
            else:
                use_linear = self._use_linear(S, self.L0)
            print("_use_linear({S}, L={L0}): {use_linear}".format(
                S=S, L0=self.L0, use_linear=use_linear))

    def _multiple_of_L0_covering_topk(self, topk):
        # Given topk, decide a nice L (candidate length)
        # Currently, it returns a multiple of #avg_poslist,
        # which can cover topk
        avglen_poslist = self.L0

        # If L is too large, set it as N (but it will too slow)
        return min((topk // avglen_poslist + 1) * avglen_poslist, self.N)

    def _use_linear(self, len_target_ids, L):
        thre = self.threshold(L)
        if len_target_ids <= thre:
            return True
        else:
            return False

    def _resolve_update_posting_lists_flag(self, flag):
        # If flag == auto, return True or False depending on nlist.
        # Otherwise, return directly
        assert flag in ["auto", True, False]
        if flag == 'auto':
            if 0 < self.nlist:
                return True
            else:
                return False
        else:
            return flag


def estimate_best_threshold_function(e, queries):
    import time
    topk = 1  # We suppose topk does not affect the result, so use topk=1

    def run(e, queries, topk, tids, L, method):
        t0 = time.time()
        for q in queries:
            if method == "linear":
                e.impl_cpp.query_linear(q, topk, tids)
            elif method == "ivf":
                e.impl_cpp.query_ivf(q, topk, tids, L)
        duration = (time.time() - t0) / queries.shape[0]  # sec/query
        return duration

    def sweep(e, queries, L):
        # Return the threshold (|S|) such that the comp. cost of query_linear and
        # query_ivf becomes the same.

        # We suppose that, if N <=128, linear is always faster
        if e.N <= 128:
            return e.N

        # Setup possible target_ids, such as [128, 256, .., 524288, 1000000], if N=10^6
        sids = [128]
        while sids[-1] * 2 < e.N:
            sids.append(sids[-1] * 2)
        sids.append(e.N)  # The last one is N

        for s in sids:
            # Approximately evaluate the runtime with just three queries
            t_linear0 = run(e=e, queries=queries[:3], L=L, method='linear', topk=topk, tids=np.arange(s))
            t_ivf0 = run(e=e, queries=queries[:3], L=L, method='ivf', topk=topk, tids=np.arange(s))

            # If linear scan gets slower than ivf
            if t_ivf0 < t_linear0:
                if s == 128:
                    if e.verbose:
                        print("ivf is faster than linear scan even if |S|<=128. This is a bit weird. "\
                              "Anyway let's set threshold as 128")
                    return 128

                # Do binary search 5 times
                s0, s1 = int(s / 2), s
                for _ in range(5):
                    s_mid = int(np.round((s0 + s1) / 2))
                    t_linear_mid = run(e=e, queries=queries, L=L, method='linear', topk=topk, tids=np.arange(s_mid))
                    t_ivf_mid = run(e=e, queries=queries, L=L, method='ivf', topk=topk, tids=np.arange(s_mid))
                    if t_ivf_mid < t_linear_mid:
                        s1 = s_mid
                    else:
                        s0 = s_mid
                return s0

        # linear is always faster
        return e.N

    if e.verbose:
        print("===== Threshold selection ====")
    xs = []
    ys = []
    for L in [k * e._multiple_of_L0_covering_topk(k) for k in [1, 2, 4, 8, 16]]:
        if e.N < L:
            continue
        thre = sweep(e=e, queries=queries, L=L)
        xs.append(L)
        ys.append(thre)

        if ys[-1] == e.N:  # Seems linear is always faster
            break

    # 1D line fitting
    if len(xs) == 1:  # If only one point available, cannot fit, so simply return the scalar
        z = [0, ys[0]]
    else:  # Otherwise, fit a line
        z = np.polyfit(xs, ys, 1)
    # Construct a 1D function in the form: p = z[0] * in + z[1]
    p = np.poly1d(z)  # new thre can be computed by thre = p(L)
    if e.verbose:
        print("L:", xs)
        print("threshold:", ys)
        print("polyfit coeff:", z)
        print("resultant func:", p)

    return p




