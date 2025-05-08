Changelog
=============

See `releases <https://github.com/matsui528/rii/releases>`_ for the latest changes.

v0.2.12 (May 8, 2025)
----------------------------
- `#67 <https://github.com/matsui528/rii/pull/67>`_ Support Python 3.12.


v0.2.11 (November 22, 2023)
----------------------------
- Support ARM CPUs.
- Further clean up the build system.
- `See here for a list of PRs <https://github.com/matsui528/rii/releases/tag/v0.2.11>`_

v0.2.10 (October 27, 2023)
----------------------------
- Update the build system to support handling the latest pip.
- `See here for a list of PRs <https://github.com/matsui528/rii/releases/tag/v0.2.10>`_


v0.2.9 (August 06, 2022)
----------------------------
- `#45 <https://github.com/matsui528/rii/pull/45>`_ Support for modern Windows CI (windows-2022). Support Python 3.9 and 3.10. Drop the support for Python 3.5.


v0.2.8 (September 12, 2021)
----------------------------
- `#42 <https://github.com/matsui528/rii/pull/42>`_, `#43 <https://github.com/matsui528/rii/pull/43>`_ Support Windows


v0.2.7 (February 4, 2020)
----------------------------
- `#36 <https://github.com/matsui528/rii/pull/36>`_ Support Mac OS (with both gcc and clang)


v0.2.6 (August 1, 2019)
----------------------------
- `#33 <https://github.com/matsui528/rii/pull/33>`_ Bug fix to install pybind11 without error


v0.2.5 (August 1, 2019)
----------------------------
- `#32 <https://github.com/matsui528/rii/pull/32>`_ Decouple pqkmeans submodule


v0.2.4 (November 25, 2018)
----------------------------
- `#29 <https://github.com/matsui528/rii/pull/29>`_ Bug fix


v0.2.3 (September 14, 2018)
----------------------------
- `#21 <https://github.com/matsui528/rii/pull/21>`_ Bug fix
- `#22 <https://github.com/matsui528/rii/pull/22>`_ Optimization for search


v0.2.2 (August 31, 2018)
----------------------------
- `#14 <https://github.com/matsui528/rii/pull/14>`_ Build on Mac with clang (without OpenMP)
- `#16 <https://github.com/matsui528/rii/pull/16>`_ SIMD implementation for squared L2 distance (SSE, AVX, and AVX512)
- `#18 <https://github.com/matsui528/rii/pull/18>`_ Implemented a merge function
- `#20 <https://github.com/matsui528/rii/pull/20>`_ Bug fix

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
