Tips
======



.. _guideline_for_search:

Guideline for search
-----------------------------

Some useful tips for tuning of search parameters:

- **Need more accurate search results**:

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

  - Try a smaller ``L`` such as ``e.query(q=q, L=int(e.L0 / 2))``.
    This is not strongly recommended because the accuracy gets worse).


.. _sequential_add:

Adding vectors sequentially
--------------------------------------------------------

You might want to add vectors one by one.
There are two ways to achieve that.

The first option is simply calling :func:`rii.Rii.add_configure` everytime.

.. code-block:: python

    # Suppose X is a set of vectors (np.ndarray with the shape (N, D))
    e = rii.Rii(fine_quantizer=codec)
    for x in X:
        e.add_configure(vecs=x.reshape(1, -1))  # Don't forget reshaping (D, ) to (1, D)

This works perfectly.
But this would take time if you would like to add many vectors by this way.
It is because the :func:`rii.Rii.reconfigure` function is called
(inside :func:`rii.Rii.add_configure`) whenever a new vector ``x`` is added.
The reconfiguration step creates postings list from scratch,
that does not need to be run for every addition.


Alternatively, you can call :func:`rii.Rii.add` for each ``x`` without updating
the posting lists, and run
:func:`rii.Rii.reconfigure` finally.

.. code-block:: python

    e = rii.Rii(fine_quantizer=codec)
    for x in X:
        e.add(vecs=x.reshape(1, -1))  # Don't forget reshaping (D, ) to (1, D)
    e.reconfigure()

This is much faster. The final results from both ways are identical.
Please remember that you must call :func:`rii.Rii.reconfigure` in the final step to create posting lists.

Note that, if you receive your data in a batch way, that can be handled in the same manner:

.. code-block:: python

    # X1 is a set of vectors (batch). Xs is a set of batches.
    # You might receive Xs as a generator/iterator
    # because the whole Xs is too large to read on memory at once
    Xs = [X1, X2, X3]

    # Running "add_configure" everytime
    e1 = rii.Rii(fine_quantizer=codec)
    for X in Xs:
        e1.add_configure(vecs=X)

    # Or, you can run "add" for each batch, and finally run "reconfigure"
    e2 = rii.Rii(fine_quantizer=codec)
    for X in Xs:
        e2.add(vecs=X)
    e2.reconfigure()



Verbose flag
---------------
You can turn on/off the verbose flag via ``e.verbose = True`` or ``e.verbose = False``. The default value is
decided by the verbose flag of the codec.


Version
---------------
The version of the package can be checked via ``rii.__version__``.

