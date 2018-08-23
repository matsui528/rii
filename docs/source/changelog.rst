Changelog
=============


v0.2.1 (August 24, 2018)
----------------------------
- `#10 <https://github.com/matsui528/rii/issues/10>`_
  Properly handling sequential data addition for the first Rii construction

v0.2.0 (August 21, 2018)
----------------------------

- `#7 <https://github.com/matsui528/rii/issues/7>`_, `#8 <https://github.com/matsui528/rii/issues/8>`_:
  Change an interface of the construction of Rii drastically.
  Now users can specify their codec explicitly.

    Old:

    .. code-block:: python

        e = rii.Rii(M=32)
        e.fit(vecs=Xt)

    Now:

    .. code-block:: python

        codec = nanopq.PQ(M=32).fit(vecs=Xt)
        e = rii.Rii(fine_quantizer=codec)

- `#8 <https://github.com/matsui528/rii/issues/8>`_ Rename the function from ``add_reconfigure`` to ``add_configure``


v0.1.0 (August 12, 2018)
----------------------------

- Initial release