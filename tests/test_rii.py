from .context import rii
import unittest
import numpy as np


class TestSuite(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_construct(self):
        M, Ks, codec = 4, 20, "pq"
        e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        self.assertEqual((e.M, e.Ks, e.codec), (M, Ks, codec))
        self.assertEqual(e.verbose, True)
        e.verbose = False
        self.assertEqual(e.verbose, False)

    def test_fit(self):
        M, Ks, codec = 4, 20, "pq"
        e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e.fit(vecs=X, iter=20, seed=123)
        self.assertEqual(e.fine_quantizer.codewords.shape, (M, Ks, D/M))

        # Can be called as a chain style
        e2 = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True).fit(vecs=X, iter=20, seed=123)
        self.assertTrue(np.allclose(e.codewords, e2.codewords))

    def test_add(self):
        for codec in ["pq", "opq"]:
            M, Ks = 4, 20
            e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e.fit(vecs=X, iter=20, seed=123)
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
        for codec in ["pq", "opq"]:
            M, Ks = 4, 20
            e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e.fit(vecs=X, iter=20, seed=123)
            e.add(vecs=X, update_posting_lists=False)
            for nlist in [5, 100]:
                e.reconfigure(nlist=nlist)
                self.assertEqual(e.nlist, nlist)
                self.assertEqual(e.coarse_centers.shape, (nlist, M))
                self.assertEqual(len(e.posting_lists), nlist)
                self.assertEqual(sum([len(plist) for plist in e.posting_lists]), N)

    def test_add_reconfigure(self):
        M, Ks, codec = 4, 20, "pq"
        e1 = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e1.fit(vecs=X, iter=20, seed=123)
        e1.add_reconfigure(vecs=X, nlist=20)
        e2 = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        e2.fit(vecs=X, iter=20, seed=123)
        e2.add(vecs=X, update_posting_lists=False)
        e2.reconfigure(nlist=20)
        # The result of add_reconfigure() should be the same as that of
        # (1) add(updating_posting_lists=False) and (2) reconfigure()
        self.assertTrue(np.allclose(e1.codes, e2.codes))
        self.assertListEqual(e1.posting_lists, e2.posting_lists)

    def test_query_linear(self):
        M, Ks, codec = 4, 20, "pq"
        e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e.fit(vecs=X, iter=20, seed=123)
        e.add_reconfigure(vecs=X, nlist=20)

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
        M, Ks, codec = 20, 256, "pq"
        e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e.fit(vecs=X, iter=20, seed=123)
        e.add_reconfigure(vecs=X, nlist=20)

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
        for codec in ["pq", "opq"]:
            M, Ks = 20, 256
            e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
            N, D = 1000, 40
            X = np.random.random((N, D)).astype(np.float32)
            e.fit(vecs=X, iter=20, seed=123)
            e.add_reconfigure(vecs=X)
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
        M, Ks, codec = 10, 256, "pq"
        e1 = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e1.fit(vecs=X, iter=20, seed=123)
        e1.add_reconfigure(vecs=X)

        import pickle
        dumped = pickle.dumps(e1)
        e2 = pickle.loads(dumped)
        self.assertEqual((e1.M, e1.Ks, e1.codec, e1.threshold),
                         (e2.M, e2.Ks, e2.codec, e2.threshold))

        self.assertTrue(np.allclose(e1.coarse_centers, e2.coarse_centers))
        self.assertTrue(np.allclose(e1.codes, e2.codes))
        for pl1, pl2 in zip(e1.posting_lists, e2.posting_lists):
            self.assertListEqual(pl1, pl2)

    def test_clear(self):
        M, Ks, codec = 4, 20, "pq"
        e = rii.Rii(M=M, Ks=Ks, codec=codec, verbose=True)
        N, D = 1000, 40
        X = np.random.random((N, D)).astype(np.float32)
        e.fit(vecs=X, iter=20, seed=123)
        e.add_reconfigure(vecs=X)
        e.clear()
        self.assertTrue(e.threshold is None)
        self.assertEqual(e.N, 0)
        self.assertEqual(e.nlist, 0)
        self.assertEqual(e.coarse_centers, None)
        self.assertEqual(e.codes, None)
        self.assertEqual(len(e.posting_lists), 0)


if __name__ == '__main__':
    unittest.main()
