<img src="http://yusukematsui.me/project/rii/img/logotype97.png" width="300">


[![Build Status](https://travis-ci.org/matsui528/rii.svg?branch=master)](https://travis-ci.org/matsui528/rii)
[![Documentation Status](https://readthedocs.org/projects/rii/badge/?version=latest)](https://rii.readthedocs.io/en/latest/?badge=latest)






Reconfigurable Inverted Index (Rii): fast and memory efficient approximate nearest neighbor search method
with a subset-search functionality.

Reference:
- [Y. Matsui](http://yusukematsui.me/), [R. Hinami](http://www.satoh-lab.nii.ac.jp/member/hinami/), and [S. Satoh](http://research.nii.ac.jp/~satoh/index.html), "**Reconfigurable Inverted Index**", ACM Multimedia 2018 (oral). [**[paper](https://arxiv.org/pdf/1808.03969)**] [**[project](http://yusukematsui.me/project/rii/rii.html)**]

## Summary of features
![](http://yusukematsui.me/project/rii/img/teaser1.png)  |  ![](http://yusukematsui.me/project/rii/img/teaser2.png)
:---:|:---:
The search can be operated for a subset of a database. | Rii remains fast even after many new items are added.
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

## Usage

### Basic ANN

```python
import rii
import nanopq
import numpy as np

N, Nt, D = 10000, 1000, 128
X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be searched
Xt = np.random.random((Nt, D)).astype(np.float32)  # 1,000 128-dim vectors for training
q = np.random.random((D,)).astype(np.float32)  # a 128-dim vector

# Prepare a PQ/OPQ codec with M=32 sub spaces
codec = nanopq.PQ(M=32).fit(vecs=Xt)  # Trained using Xt

# Instantiate a Rii class with the codec
e = rii.Rii(fine_quantizer=codec)

# Add vectors
e.add_configure(vecs=X)

# Search
ids, dists = e.query(q=q, topk=3)
print(ids, dists)  # e.g., [7484 8173 1556] [15.06257439 15.38533878 16.16935158]
```
Note that, if you want, you can construct a codec at the same time as the instantiation of the Rii class
```python
e = rii.Rii(fine_quantizer=nanopq.PQ(M=32).fit(vecs=Xt))
e.add_configure(vecs=X)
```
Furthermore, you can even construct the class and add the vectors in one line
```python
e = rii.Rii(fine_quantizer=nanopq.PQ(M=32).fit(vecs=Xt)).add_configure(vecs=X)
```

### Subset search

```python
# The search can be conducted over a subset of the database
target_ids = np.array([85, 132, 236, 551, 694, 728, 992, 1234])  # Specified by IDs
ids, dists = e.query(q=q, topk=3, target_ids=target_ids)
print(ids, dists)  # e.g., [728  85 132] [14.80522156 15.92787838 16.28690338]
```

### Data addition and reconfiguration

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

### I/O by pickling
```python
import pickle
with open('rii.pkl', 'wb') as f:
    pickle.dump(e, f)
with open('rii.pkl', 'rb') as f:
    e_dumped = pickle.load(f)  # e_dumped is identical to e
```



## Author
- [Yusuke Matsui](http://yusukematsui.me)

## Credits
- The logo is designed by [@richardbmx](https://github.com/richardbmx) ([#4](https://github.com/matsui528/rii/issues/4))