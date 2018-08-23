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

  - Try a smaller ``L`` such as ``e.query(q=q, L=e.L0 / 2)``.
    This is not strongly recommended because the accuracy gets worse).


.. _sequential_add:

Initializing a Rii class by adding vectors sequentially
--------------------------------------------------------

For the first data addition, one might want to add vectors one by one.
There are two ways to achieve that.

The first option is simply calling :func:`rii.Rii.add_configure` everytime.

.. code-block:: python

    e = rii.Rii(fine_quantizer=codec)
    for x in X:
        e.add_configure(vecs=x.reshape(1, -1))  # Don't forget reshaping (D, ) to (1, D)

This works perfectly. But this would take time if you would like to add many vectors
by this way.
It is because the reconfigure function is called (i.e., posting lists are computed from
scrath) whenever each vector ``x`` is added.

Alternatively, you can call :func:`add` for each ``x`` without updating
the posting lists, and run
:func:`reconfigure` finally.

.. code-block:: python

    e = rii.Rii(fine_quantizer=codec)
    for x in X:
        e.add(vecs=x.reshape(1, -1))  # Don't forget reshaping (D, ) to (1, D)
    e.reconfigure()

This is much faster. The final result of both ways are same.
But you must call :func:`rii.Rii.reconfigure` in the final step to create posting lists.



Verbose flag
---------------
You can turn on/off the verbose flag via ``e.verbose = True`` or ``e.verbose = False``. The default value is
decided by the verbose flag of the codec.


