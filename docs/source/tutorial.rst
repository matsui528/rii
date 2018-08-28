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
The package is tested on linux with g++.
It should work on Mac with clang (without OpenMP) as well, but not fully tested.




Basic of Rii
------------


Let us first prepare 10,000 128-dim database vectors to be indexed,
1,000 128-dim vectors for training,
and a 128-dim query vector. They must be np.ndarray with np.float32.
Our objective is to find similar vectors to the query from the database vectors
efficiently.

.. code-block:: python

    import rii
    import nanopq
    import numpy as np

    N, Nt, D = 10000, 1000, 128
    X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be searched
    Xt = np.random.random((Nt, D)).astype(np.float32)  # 1,000 128-dim vectors for training
    q = np.random.random((D,)).astype(np.float32)  # a 128-dim vector

First, a PQ/OPQ codec
(`nanopq.PQ <https://nanopq.readthedocs.io/en/latest/source/api.html#product-quantization-pq>`_ or
`nanopq.OPQ <https://nanopq.readthedocs.io/en/latest/source/api.html#optimized-product-quantization-opq>`_)
needs to be prepared.
This codec will be used to encode/decode vectors.
Note that PQ refers to the usual Product Quantization [Jegou11]_.
OPQ means Optimized Product Quantization [Ge14]_, which is an extended version of PQ.
Compared to PQ, OPQ is little bit slower for encoding/searching but slightly more accurate.


.. code-block:: python

    # Prepare a PQ/OPQ codec with M=32 sub spaces
    codec = nanopq.PQ(M=32, Ks=256, verbose=True).fit(vecs=Xt)  # Trained using Xt

See `the tutorial of nanopq <https://nanopq.readthedocs.io/en/latest/source/tutorial.html>`_
for more details about the parameter selection of the codec.
Note that you can use ``X`` or the part of ``X`` for training if you
cannot prepare training vectors ``Xt``. For example: ``codec = nanopq.PQ(M=32).fit(vecs=X[:1000])``

Let's instantiate a search class, :class:`rii.Rii`, with the trained codec as a fine quantizer:

.. code-block:: python

    # Instantiate a Rii class with the codec
    e = rii.Rii(fine_quantizer=codec)

The codec will be stored in :attr:`rii.Rii.fine_quantizer`.
You can see its codewords via :attr:`rii.Rii.codewords`.

Next, let us add the vectors ``X`` into the rii instance
by running :func:`rii.Rii.add_configure`:

.. code-block:: python

    # Add vectors
    e.add_configure(vecs=X, nlist=None, iter=5)

Inside this function, :func:`rii.Rii.add` and :func:`rii.Rii.reconfigure` are called:

- :func:`rii.Rii.add`

  - The input vectors ``X`` are encoded to memory-efficient PQ-codes via :attr:`rii.Rii.fine_quantizer`.
    See `the tutorial of nanopq <https://nanopq.readthedocs.io/en/latest/source/tutorial.html>`_
    for more details about PQ encoding.

  - The resultant PQ-codes are stored in the Rii instance.
    Note that you can access them via :attr:`rii.Rii.codes`.

- :func:`rii.Rii.reconfigure`

  - For the fast search, an inverted index structure is created by this function.

  - The PQ-codes are groupted into several clusters via PQk-means [Matsui17]_.
    You can access the resultant cluster centers via :attr:`rii.Rii.coarse_centers`.
    The assignment for each PQ-code to its nearest center is stored on :attr:`rii.Rii.posting_lists`.

  - The number of centers is denoted by the parameter ``nlist``.
    The default value is None, where ``nlist`` is set to ``sqrt(N)`` automatically
    as suggested `here <https://github.com/facebookresearch/faiss/wiki/Index-IO,-index-factory,-cloning-and-hyper-parameter-tuning#guidelines>`_.
    The number of iteration for the clustering process
    is specified by ``iter``.

Make sure that you must call :func:`rii.Rii.add_configure` (not :func:`rii.Rii.add`)
for the first data addition. It is because you need to create coarse centers (posting lists).
Note that, if you would like to add vectors sequentially
when constructing the class, please refer this; :ref:`sequential_add`


.. hint::

    By the way, you can construct a codec at the same time as the instantiation of the Rii class
    if you want to write them in one line.

    .. code-block:: python

        e = rii.Rii(fine_quantizer=nanopq.PQ(M=32).fit(vecs=Xt))
        e.add_configure(vecs=X)

    Furthermore, you can even construct the class and add the vectors in the same line
    by chaining functions.

    .. code-block:: python

        e = rii.Rii(fine_quantizer=nanopq.PQ(M=32).fit(vecs=Xt)).add_configure(vecs=X)


Finally, we can run a search for a given query vector ``q``.

.. code-block:: python

    # Search
    ids, dists = e.query(q=q, topk=3, L=None, target_ids=None, sort_target_ids=True, method='auto')
    print(ids, dists)  # e.g., [7484 8173 1556] [15.06257439 15.38533878 16.16935158]


See the docstring :func:`rii.Rii.query` for the details of each parameter.
I recommend running the search with the default parameters first.
For parameter tuning, please refer
:ref:`guideline_for_search` for more details.







Subset search
-----------------

The search can be conducted on a **subset** of the whole PQ-codes.
Such subset-search is practically important. For example of image search,
we can filter out unrelated images by checking their tags, and run feature-based search
to find the similar images to the query.

A subset is specified simply by a numpy array, ``target_ids``.

.. code-block:: python

    # The search can be conducted over a subset of the database
    target_ids = np.array([85, 132, 236, 551, 694, 728, 992, 1234])  # Specified by IDs
    ids, dists = e.query(q=q, topk=3, target_ids=target_ids, sort_target_ids=True)
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


.. hint::

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


Data addition and reconfiguration
-------------------------------------

Although there exist many fast ANN algorithms,
almost all methods are optimized for an initial item set.
It is not always clear how the search performance degrades when many items are newly added.
Rii provides a **reconfigure** function, by which the search remains fast
even after many vectors are newly added.

Let us first show how to add new vectors.
Suppose a Rii instance is constructed with 10,000 items.
Given the constructed Rii instance,
you can call :func:`rii.Rii.add` to add new vectors.
The search can be conducted by :func:`rii.Rii.query`.
This works well when ``X2`` is small enough:

.. code-block:: python

    # Suppose e was constructed with 10,000 PQ-codes.

    # Add new vectors
    X2 = np.random.random((1000, D)).astype(np.float32)
    e.add(vecs=X2)  # Now N is 11000
    e.query(q=q)  # Ok. (0.12 msec / query)



However, if you add quite a lot of vectors,
the search might become slower.
It is because the data structure has been optimized for the initial items (N=10000).

.. code-block:: python

    X3 = np.random.random((1000000, D)).astype(np.float32)
    e.add(vecs=X3)  # A lot. Now N is 1011000
    e.query(q=q)  # Slower (0.96 msec/query)


In such case, you can run :func:`rii.Rii.reconfigure`.
That updates the data structure (re-computes the coarse centers and posting lsits),
making the search faster.

.. code-block:: python

    e.reconfigure(nlist=None, iter=5)
    e.query(q=q)  # Ok. (0.21 msec / query)


Note that, if you want, the above addition and reconfiguration
can be achieved at the same time with one line by
:func:`rii.Rii.add_configure`:

.. code-block:: python

    X3 = np.random.random((1000000, D)).astype(np.float32)
    e.add_configure(vecs=X3, nlist=None, iter=5)







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

There are some utility functions: :func:`rii.Rii.print_params`, :func:`rii.Rii.clear`,
and :func:`rii.Rii.merge`.

.. code-block:: python

    # Print the current parameters
    e.print_params()

    # Delete all PQ-codes and posting lists. fine_quantizer is kept.
    e.clear()

    # You can merge two Rii instances if they have the same fine_quantizer
    e1 = rii.Rii(fine_quantizer=codec)
    e2 = rii.Rii(fine_quantizer=codec)
    e1.add_reconfigure(vecs=X1)
    e2.add_reconfigure(vecs=X2)
    e1.merge(e2)  # e1 will have (PQ-codes of) both X1 and X2




More examples
-----------------

See more advanced examples as follows

- `Simple tag search <https://github.com/matsui528/rii/tree/master/examples/tag_search/simple_tag_search.ipynb>`_






.. [Jegou11] H. Jegou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
.. [Ge14] T. Ge, K. He, Q. Ke, and J. Sun, "Optimized Product Quantization", IEEE TPAMI 2014
.. [Matsui17] Y. Matsui, K. Ogaki, T. Yamasaki, and K. Aizawa, "PQk-means: Billion-scale Clustering for Product-quantized Codes", ACM Multimedia 2017