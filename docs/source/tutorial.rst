Tutorial
==========




This tutorial shows the basic usage of Reconfigurable Inverted Index (*Rii*) [Matsui18]_;
a fast and memory-efficient approximate nearest neighbor search method.

Rii has two key features:

- **Subset search**: Rii enables efficient search
  on a **subset** of the whole database, regardless of the size of the subset.
- **Reconfiguration**: Rii remains fast, even after many new items are added,
  because the data structure is dynamically adjusted for the current
  number of database items.

Based on the well-known inverted file
with product quantization (PQ) approach (so called **IVFADC** or **IVFPQ**) [Jegou11]_,
the data layout of Rii is designed such that an item can be fetched
by its identifier with a cost of O(1).
This simple but critical modification enables us to search over a subset of the dataset efficiently
by switching to a linear PQ scan if the size of the subset is small.
Owing to this linear layout, the granularity of a coarse assignment step can easily be controlled
by running clustering again over the dataset whenever the user wishes.
This means that the data structure can be adjusted dynamically
after new items are added.


Th core part of Rii is implemented in C++11, and bound to the Python interface
by `pybind11 <https://github.com/pybind/pybind11>`_.
The search is automatically parallelized by `OpenMP <https://www.openmp.org/>`_.


.. Note that the advanced encoding, Optimized Product Quantization (OPQ) [Ge14]_ is also supported.


Basic of Rii
------------


Let us first prepare 10,000 128-dim database vectors to be indexed,
and a 128-dim query vector. They must be np.ndarray with np.float32.

.. code-block:: python

    import rii
    import numpy as np

    N, D = 10000, 128
    X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors
    q = np.random.random((D,)).astype(np.float32)  # a 128-dim vector

Let's instantiate a :class:`rii.Rii` class.

.. code-block:: python

    # Instantiate with M=32 sub-spaces
    e = rii.Rii(M=32, Ks=256, codec='pq', verbose='False')

Each vector will be splitted into ``M`` sub-vectors (so ``D`` must be dividable by ``M``),
and quantized with ``Ks`` codewords.
Note that ``M`` is a parameter to control the trade off of accuracy and memory-cost.
If you set larger ``M``, you can achieve better quantization (i.e., less reconstruction error)
with more memory usage.
``Ks`` specifies the number of codewords for quantization.
This is tyically 256 so that each sub-space is represented by 8 bits = 1 byte = np.uint8.
The memory cost for each pq-code is ``M`` :math:`* \log_2` ``Ks`` bits.

You can select an encode/decode method from `pq` or `opq`.
The usual Product Quantization [Jegou11]_ is specified by `pq`.
Optimized Product Quantization [Ge14]_, which is little bit slower for
encoding/searching but slightly more accurate, is specified by `opq`.
The codec is stored as :attr:`rii.Rii.fine_quantizer`. This is an instance of
`nanopq.PQ <https://nanopq.readthedocs.io/en/latest/source/api.html#product-quantization-pq>`_
or `nanopq.OPQ <https://nanopq.readthedocs.io/en/latest/source/api.html#optimized-product-quantization-opq>`_.

Next, the codewords of PQ/OPQ are trained by :func:`rii.Rii.fit` using some training vectors, with ``iter`` iterations and
``seed`` for the seed of random process.

.. code-block:: python

    # Train with the top 1000 vectors
    e.fit(vecs=X[:1000], iter=20, seed=123)


Here, the top 1,000  database vectors are used for training.
You can prepare other training vectors for this step if you cannot prepare
the database vectors.
You can see the resulting codewords via :attr:`rii.Rii.codewords`.
Note that, alternatively, you can instantiate and train the instance in one line if you want:

.. code-block:: python

    e = rii.Rii(M=32, Ks=256, codec='pq', verbose='False').fit(vecs=X[:1000], iter=20, seed=123)


Next, let us add the vectors ``X`` into the rii instance.

.. code-block:: python

    # Add vectors
    e.add_reconfigure(vecs=X, nlist=None, iter=5)

Inside :func:`rii.Rii.add_reconfigure`, all vectors are converted to PQ-codes
and stored (:func:`rii.Rii.add`).
The PQ-codes are then grouped into ``nlist`` clusters (:func:`rii.Rii.reconfigure`),
i.e., :attr:`rii.Rii.coarse_centers` are computed for the coarse assignment step.
The default value of ``nlist`` is None, by which ``nlist`` is set to :math:`\sqrt{N}`
as suggested `here <https://github.com/facebookresearch/faiss/wiki/Index-IO,-index-factory,-cloning-and-hyper-parameter-tuning#guidelines>`_.
The number of iteration for the clustering process
via PQk-means [Matsui17]_ is specified by ``iter``.
The resultant PQ-codes and posting lists can be accessed by :attr:`rii.Rii.codes`
and :attr:`rii.Rii.posting_lists`, respectively.

Note that you must call :func:`rii.Rii.add_reconfigure` (not :func:`rii.Rii.add`)
if you first add vectors because you need to create coarse centers.


Finally, we can run a search for a given query vector ``q``.

.. code-block:: python

    # Search
    ids, dists = e.query(q=q, topk=3, L=None, target_ids=None, sort_target_ids=True, method='auto')
    print(ids, dists)  # e.g., [7484 8173 1556] [15.06257439 15.38533878 16.16935158]


See the docstring :func:`rii.Rii.query` for the details of each parameter.
You can first run the search with default parameters.
For parameter tuning, please see
:ref:`guideline_for_search` for more details.







Subset search
-----------------

The search can be conducted on a **subset** of the whole PQ-codes.
Such subset-search is practically important, for example of image search,
we can filter out unrelated images by checking their tags, and run feature-based search
to find the similar images to the query.

A subset is specified simply by a numpy array, ``target_ids``.

.. code-block:: python

    # The search can be conducted over a subset of the database
    target_ids = np.array([85, 132, 236, 551, 694, 728, 992, 1234])  # Specified by IDs
    ids, dists = e.query(q=q, topk=3, target_ids=target_ids, sort_target_ids=False)
    print(ids, dists)  # e.g., [728  85 132] [14.80522156 15.92787838 16.28690338]

As can be seen in the resulted identifiers ``ids``, the search result includes
the items specified by ``target_ids`` only. Note that:

- Make sure ``target_ids`` must be np.ndarray with ``ndim=1`` and ``dtype=np.int64``.

- Please don't include duplicate identifiers in ``target_ids``. The behavior is undefined.

- The target identifiers must be sorted before the search (see Sec 4.2 in [Matsui18]_ for details).
  In a default setting, ``sort_target_ids`` is True. This means that
  ``target_ids`` will be sorted inside the query function, so you do not need to
  manually sort ``target_ids`` before running :func:`rii.Rii.query`.
  This works practically well when ``target_ids`` is not so large.

- If ``target_ids`` contans a lot of identifiers,
  sorting could become slower than the search itself.
  In such case, you can manually sort ``target_ids`` beforehand, and
  pass it to :func:`rii.Rii.query` with ``sort_target_ids=False``.
  This is a complete procedure explained in the paper.


Some examples of subset-search are:

.. code-block:: python

    # Because target ids are not sorted, sort_target_ids must be True (default behavior)
    e.query(q=q, topk=1, target_ids=np.array([345, 23, 994, 425]))

    # The search is run on the 1st to 1000th items.
    # Since the target_ids are already sorted, you can set False for the sort flag.
    e.query(q=q, topk=1, target_ids=np.arange(1000), sort_target_ids=False)

    # Search for several queries with a large target_ids. In such case,
    # it is redundant to sort inside the query function every time; you should sort only once
    target_ids = np.array([44432, 32786, ..., 9623])   # Lots of identifiers
    target_ids = np.sort(target_ids)  # Do sort
    for q in Q:
        # Here, ths sort flag is off for efficient search
        e.search(q=q, topk=1, target_ids=target_ids, sort_target_ids=False)


Reconfiguration (data addition)
-----------------------------------

Although there exist many fast ANN algorithms,
almost all methods are optimized for an initial item set.
It is not always clear how the search performance degrades when new items are added.
Rii provides a **reconfigure** function, by which the search remains fast
even after many vectors are newly added.

Let us first show how to add new vectors.

.. code-block:: python

    # Suppose e has 10,000 PQ-codes.

    # Add new vectors
    X2 = np.random.random((1000, D)).astype(np.float32)
    e.add(vecs=X2)  # Now N is 11000
    e.query(q=q)  # Ok. (0.12 msec / query)

You can call :func:`rii.Rii.add` to add new vectors.
The search can be conducted by :func:`rii.Rii.query`.
This works well when ``X2`` is small enough.


However, if you add quite a lot of vectors,
the search might become slower
because the data structure has been optimized for the initial items (N=10000).

.. code-block:: python

    X3 = np.random.random((1000000, D)).astype(np.float32)
    e.add(vecs=X3)  # A lot. Now N is 1011000
    e.query(q=q)  # Slower (0.96 msec/query)


In such case, you can run :func:`rii.Rii.reconfigure`.
That updates the data structure, making the search faster.

.. code-block:: python

    e.reconfigure(nlist=None, iter=5)
    e.query(q=q)  # Ok. (0.21 msec / query)


Note that, if you add several items in a batch manner,
you can skip to update posting lists until the final reconfigure.

.. code-block:: python

    # Batch addition example. Suppose there are big matrices Xa, Xb, ...
    for filename in ["Xa.npy", "Xb.npy", "Xc.npy"]:
        X = np.load(filename)
        e.add(vecs=X, update_posting_lists=False)
    e.reconfigure()

This produces exact the same results with ``e.add(vecs=X)``, but faster.




I/O by pickling
------------------

The rii class supports pickling. You can read/write an instance easily.

.. code-block:: python

    import pickle

    with open('rii.pkl', 'wb') as f:
        pickle.dump(e, f)

    with open('rii.pkl', 'rb') as f:
        e_dumped = pickle.load(f)  # e_dumped is identical to e



Utility functions
-----------------

There are two utility functions, :func:`rii.Rii.print_params` and :func:`rii.Rii.clear`.

.. code-block:: python

    # Print the current parameters
    e.print_params()

    # Delete all PQ-codes and posting lists. fine_quantizer is kept.
    e.clear()




.. _guideline_for_search:

Guideline for search
---------------------

- **Need more accurate search**:

  - Set a larger ``L`` in :func:`rii.Rii.query`.
    This usually improves the accuracy, but makes the search slower.
    The recommended way is to set a multiple of :attr:`rii.Rii.L0`, e.g.,
    ``e.query(q=q, L=4 * e.L0)``

  - If you find changing ``L`` is not enough, please construct the rii with
    a larger ``M`` value.
    Again, this change boosts the accuracy, but the runtime becomes slower.
    Don't forget that the dimensionality of the vector ``D`` must be dividable by ``M``.

  - If your codec is `pq`, please consider to switch to `opq`.

- **Want to make the search faster**:

  - Please run :func:`rii.Rii.reconfigure` with a larger ``nlist``, such as
    ``e.reconfigure(nlist=4*np.sqrt(e.N))``

  - If your task is subset-search, please consider sorting ``target_ids`` before
    passing it to the query function, and call :func:`rii.Rii.query` with
    ``sort_target_ids=False``.

  - Try a smaller ``L`` such as ``e.query(q=q, L=e.L0 / 2)``.
    This is not strongly recommended because the accuracy gets worse).








.. [Jegou11] H. Jegou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
.. [Ge14] T. Ge, K. He, Q. Ke, and J. Sun, "Optimized Product Quantization", IEEE TPAMI 2014
.. [Matsui17] Y. Matsui, K. Ogaki, T. Yamasaki, and K. Aizawa, "PQk-means: Billion-scale Clustering for Product-quantized Codes", ACM Multimedia 2017