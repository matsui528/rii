from .context import rii
import unittest
import numpy as np
import nanopq


class TestSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_construct(self):
        M, Ks = 4, 20
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        self.assertEqual(e.fine_quantizer.codewords.shape, (M, Ks, D/M))
        self.assertEqual((e.M, e.Ks), (M, Ks))
        self.assertEqual(e.verbose, True)
        e.verbose = False
        self.assertEqual(e.verbose, False)

    def test_add(self):
        for codec in [nanopq.PQ, nanopq.OPQ]:
            M, Ks = 4, 20
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e = rii.Rii(fine_quantizer=codec(M=M, Ks=Ks, verbose=True).fit(vecs=X))

            self.assertEqual(e.N, 0)
            e.add(vecs=X, update_posting_lists=False)
            self.assertEqual(e.N, N)

            # The encoded vectors should be equal to the ones manually PQ-encoded
            pq = e.fine_quantizer
            codes = pq.encode(X)
            self.assertTrue(np.allclose(codes, e.codes))

            # Add again
            e.add(vecs=X, update_posting_lists=False)
            self.assertEqual(e.N, 2 * N)

    def test_reconfigure(self):
        for codec in [nanopq.PQ, nanopq.OPQ]:
            M, Ks = 4, 20
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e = rii.Rii(fine_quantizer=codec(M=M, Ks=Ks, verbose=True).fit(vecs=X))
            e.add(vecs=X, update_posting_lists=False)
            for nlist in [5, 100]:
                e.reconfigure(nlist=nlist)
                self.assertEqual(e.nlist, nlist)
                self.assertEqual(e.coarse_centers.shape, (nlist, M))
                self.assertEqual(len(e.posting_lists), nlist)
                self.assertEqual(sum([len(plist) for plist in e.posting_lists]), N)

    def test_add_configure(self):
        M, Ks = 4, 20
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e1 = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e1.add_configure(vecs=X, nlist=20)
        e2 = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e2.add(vecs=X, update_posting_lists=False)
        e2.reconfigure(nlist=20)
        # The result of add_configure() should be the same as that of
        # (1) add(updating_posting_lists=False) and (2) reconfigure()
        self.assertTrue(np.allclose(e1.codes, e2.codes))
        self.assertListEqual(e1.posting_lists, e2.posting_lists)
        e3 = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X)).add_configure(vecs=X, nlist=20)
        # Can be called as a chain
        self.assertTrue(np.allclose(e1.codes, e3.codes))
        self.assertListEqual(e1.posting_lists, e3.posting_lists)

    def test_add_configure_small_number_of_vectors(self):
        import copy
        M, Ks = 4, 20
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e1 = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e2 = copy.deepcopy(e1)
        e3 = copy.deepcopy(e1)
        for x in X[:10]:
            e1.add_configure(vecs=x.reshape(1, -1))  # Can be added one by one
        self.assertEqual(e1.N, 10)

        e2.add_configure(vecs=X[:10])
        # Should be same to that by add_reconfigure at once
        self.assertTrue(np.allclose(e1.codes, e2.codes))
        self.assertEqual(e1.posting_lists, e2.posting_lists)

        for x in X[:10]:
            e3.add(x.reshape(1, -1))
        e3.reconfigure()
        # Should be same to that by add several times the nreconfigure
        self.assertTrue(np.allclose(e1.codes, e3.codes))
        self.assertEqual(e1.posting_lists, e3.posting_lists)



    def test_query_linear(self):
        M, Ks = 4, 20
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e.add_configure(vecs=X, nlist=20)

        for n, q in enumerate(X[:10]):
            topk = 10
            ids1, dists1 = e.impl_cpp.query_linear(q, topk, np.array([], dtype=np.int64))
            self.assertTrue(isinstance(ids1, list))
            self.assertTrue(isinstance(ids1[0], int))
            self.assertTrue(isinstance(dists1, list))
            self.assertTrue(isinstance(dists1[0], float))
            self.assertEqual(len(ids1), topk)
            self.assertEqual(len(ids1), len(dists1))
            self.assertTrue(np.all(0 <= np.diff(dists1)))  # Make sure dists1 is sorted
            # The true NN is included in top 10 with high prob
            self.assertTrue(n in ids1)

            # Subset search w/ a full indices should be the same w/o target
            ids2, dists2 = e.impl_cpp.query_linear(q, topk, np.arange(N))
            self.assertListEqual(ids1, ids2)
            self.assertListEqual(dists1, dists2)

            S = np.array([2, 24, 43, 55, 102, 139, 221, 542, 667, 873, 874, 899])
            ids3, dists3 = e.impl_cpp.query_linear(q, topk, S)
            self.assertTrue(np.all([id in S for id in ids3]))

    def test_query_ivf(self):
        M, Ks = 20, 256
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e.add_configure(vecs=X, nlist=20)

        for n, q in enumerate(X[:10]):
            L = 200
            topk = 10
            ids1, dists1 = e.impl_cpp.query_ivf(q, topk, np.array([], dtype=np.int64), L)
            self.assertTrue(isinstance(ids1, list))
            self.assertTrue(isinstance(ids1[0], int))
            self.assertTrue(isinstance(dists1, list))
            self.assertTrue(isinstance(dists1[0], float))
            self.assertEqual(len(ids1), topk)
            self.assertEqual(len(ids1), len(dists1))
            self.assertTrue(np.all(0 <= np.diff(dists1)))  # Make sure dists1 is sorted
            # The true NN is included in top 10 with high prob
            # This might fail if the parameters are severe
            self.assertTrue(n in ids1)

            # Subset search w/ a full indices should be the same w/o target
            ids2, dists2 = e.impl_cpp.query_ivf(q, topk, np.arange(N), L)
            self.assertListEqual(ids1, ids2)
            self.assertListEqual(dists1, dists2)

            S = np.array([2, 24, 43, 55, 102, 139, 221, 542, 667, 873, 874, 899])
            ids3, dists3 = e.impl_cpp.query_ivf(q, topk, S, L)
            self.assertTrue(np.all([id in S for id in ids3]))

            # When target_ids is all vectors and L=all, the results is the same as linear PQ scan
            ids4, dists4 = e.impl_cpp.query_ivf(q, topk, np.arange(N), N)
            ids5, dists5 = e.impl_cpp.query_linear(q, topk, np.array([], dtype=np.int64))
            self.assertListEqual(ids4, ids5)
            self.assertListEqual(dists4, dists5)

            # When target_ids is specified and L is large, linear and ivf should produce the same result
            ids6, dists6 = e.impl_cpp.query_ivf(q, topk, S, L)
            ids7, dists7 = e.impl_cpp.query_linear(q, topk, S)
            self.assertListEqual(ids6, ids7)
            self.assertListEqual(dists6, dists7)

    def test_query(self):
        for codec in [nanopq.PQ, nanopq.OPQ]:
            M, Ks = 20, 256
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e = rii.Rii(fine_quantizer=codec(M=M, Ks=Ks, verbose=True).fit(vecs=X))
            e.add_configure(vecs=X, nlist=20)

            for n, q in enumerate(X[:10]):
                topk=50
                ids1, dists1 = e.query(q=q, topk=topk)
                self.assertTrue(isinstance(ids1, np.ndarray))
                self.assertEqual(ids1.dtype, np.int64)
                self.assertTrue(isinstance(dists1, np.ndarray))
                self.assertEqual(dists1.dtype, np.float64)
                self.assertEqual(len(ids1), topk)
                self.assertEqual(len(ids1), len(dists1))
                self.assertTrue(np.all(0 <= np.diff(dists1)))  # Make sure dists1 is sorted
                # The true NN is included in top 10 with high prob
                # This might fail if the parameters are severe
                self.assertTrue(n in ids1)

                # Subset search w/ a full indices should be the same w/o target
                ids2, dists2 = e.query(q=q, topk=topk, target_ids=np.arange(N))
                self.assertTrue(np.allclose(ids1, ids2))
                self.assertTrue(np.allclose(dists1, dists2))

                S = np.array([2, 24, 43, 55, 102, 139, 221, 542, 667, 873, 874, 899])
                ids3, dists3 = e.query(q=q, topk=5, target_ids=S)
                self.assertTrue(np.all([id in S for id in ids3]))

    def test_pickle(self):
        M, Ks = 10, 256
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e1 = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e1.add_configure(vecs=X, nlist=20)

        import pickle
        dumped = pickle.dumps(e1)
        e2 = pickle.loads(dumped)
        self.assertEqual((e1.M, e1.Ks, e1.threshold),
                         (e2.M, e2.Ks, e2.threshold))

        self.assertTrue(np.allclose(e1.coarse_centers, e2.coarse_centers))
        self.assertTrue(np.allclose(e1.codes, e2.codes))
        for pl1, pl2 in zip(e1.posting_lists, e2.posting_lists):
            self.assertListEqual(pl1, pl2)

    def test_clear(self):
        M, Ks = 4, 20
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X))
        e.add_configure(vecs=X, nlist=20)
        e.clear()
        self.assertTrue(e.threshold is None)
        self.assertEqual(e.N, 0)
        self.assertEqual(e.nlist, 0)
        self.assertEqual(e.coarse_centers, None)
        self.assertEqual(e.codes, None)
        self.assertEqual(len(e.posting_lists), 0)

    def test_merge(self):
        from itertools import chain
        M, Ks, N1, N2, D = 4, 20, 1000, 500, 40
        X1 = np.random.random((N1, D)).astype(np.float32)
        X2 = np.random.random((N2, D)).astype(np.float32)
        codec = nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X1)
        e1 = rii.Rii(fine_quantizer=codec)
        e2 = rii.Rii(fine_quantizer=codec)

        # e1: empty  e2: empty
        e1.merge(e2)
        self.assertEqual((e1.N, e2.N), (0, 0))

        # e1: vecs  e2: empty
        e1.add_configure(vecs=X1)
        e1.merge(e2)  # posting lists are created in the above line
        self.assertEqual(e1.N, N1)
        self.assertEqual(e1.nlist, int(np.sqrt(N1)))  # Have posting lists
        e1.clear()

        # e1: empty  e2: vecs
        e2.add_configure(vecs=X2)
        e1.merge(e2)  # e1 didn't have posting lists
        self.assertEqual(e1.N, N2)
        self.assertEqual(e1.nlist, 0)  # No posting lists
        e1.clear()
        e2.clear()

        # e1: vecs  e2: vecs
        e1.add_configure(vecs=X1)
        e2.add_configure(vecs=X2)
        e1.merge(e2)
        self.assertEqual(e1.N, N1 + N2)
        self.assertEqual(e1.nlist, int(np.sqrt(N1)))  # posting lists are same as the original e1

        # Make sure everything is fine
        self.assertTrue(np.array_equal(e1.codes, codec.encode(np.vstack((X1, X2)))))
        self.assertEqual(sorted(chain(*e1.posting_lists)), list(range(N1 + N2)))

    ### For debugging ###
    # def test_runtime(self):
    #     import time
    #     M, Ks, N, D = 8, 256, 100000, 128
    #     X = np.random.random((N, D)).astype(np.float32)
    #     e = rii.Rii(fine_quantizer=nanopq.PQ(M=M, Ks=Ks, verbose=True).fit(vecs=X[:1000])).add_configure(vecs=X)
    #     Q = np.random.random((10000, D)).astype(np.float32)
    #     t0 = time.time()
    #     for q in Q:
    #         e.query(q=q, topk=3, method='ivf')
    #     print(time.time() - t0, "sec")

if __name__ == '__main__':
    unittest.main()
