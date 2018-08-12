# Rii

[![Build Status](https://travis-ci.org/matsui528/rii.svg?branch=master)](https://travis-ci.org/matsui528/rii)
[![Documentation Status](https://readthedocs.org/projects/rii/badge/?version=latest)](https://rii.readthedocs.io/en/latest/?badge=latest)

Reconfigurable Inverted Index (Rii): fast and memory efficient approximate nearest neighbor search method
with a subset-search functionality.

Reference:
- Y. Matsui, R. Hinami, and S. Satoh, "**Reconfigurable Inverted Index**", ACM Multimedia 2018 (oral). [[paper]()] [[supplementary]()] [[project](http://yusukematsui.me/project/rii/rii.html)]

## Summary of features
- Fast and memory efficient ANN. Can handle billion-scale data on memory at once. The search is less than 10 ms.
- Can run the search over a **subset** of the whole database
- Remain fast even after a large number of vectors are newly added (i.e., the data structure can be **reconfigured**)


## Installing
You can install the package via pip. This library works with Python 3.5+ on linux.
```
pip install rii
```

## [Documentation](https://rii.readthedocs.io/en/latest/index.html)
- [Tutorial](https://rii.readthedocs.io/en/latest/source/tutorial.html)
- [API](https://rii.readthedocs.io/en/latest/source/api.html)

## Example

### Basic ANN

```python
import rii
import numpy as np

N, D = 10000, 128
X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors
q = np.random.random((D,)).astype(np.float32)  # a 128-dim vector

# Instantiate with M=32 sub-spaces
e = rii.Rii(M=32)

# Train with the top 1000 vectors
e.fit(vecs=X[:1000])

# Add vectors
e.add_reconfigure(vecs=X)

# Search
ids, dists = e.query(q=q, topk=5)
print(ids, dists)  # e.g., [1965  951 1079] [11.77537251 13.04397392 13.06065941]
```

### Subset search

```python
# The search can be conducted over a subset of the database
target_ids = np.array([85, 132, 236, 551, 694, 728, 992, 1234])  # Specified by IDs
ids, dists = e.query(q=q, topk=3, target_ids=target_ids)
print(ids, dists)  # e.g., [728  85 132] [14.80522156 15.92787838 16.28690338]
```

### Reconfigure

```python
# Add new vectors
X2 = np.random.random((1000, D)).astype(np.float32)
e.add(vecs=X2)  # Now N is 11000
e.query(q=q)  # Ok. (0.12 msec / query)

# However, if you add quite a lot of vectors, the search might become slower
# because the data structure has been optimized for the initial item size (N=10000)
X3 = np.random.random((1000000, D)).astype(np.float32) 
e.add(vecs=X3)  # A lot. Now N is 1011000
e.query(q=q)  # Slower (0.96 msec/query)

# In such case, run the reconfigure function. That updates the data structure
e.reconfigure()
e.query(q=q)  # Ok. (0.21 msec / query)
```



## Author
- [Yusuke Matsui](http://yusukematsui.me)

## Citation

    @inproceedings{rii,
      author = {Yusuke Matsui and Ryota Hinami and Shin'ichi Satoh},
      title = {Reconfigurable Inverted Index},
      booktitle = {ACM International Conference on Multimedia (ACMMM)},
      year = {2018},
    }


