# Benchmark

The benchmarking scripts for SIFT1M and SIFT1B.

## SIFT1M
You can compare Rii, Faiss (IVFADC), NMSLIB (HNSW), FALCONN, and Annoy, using SIFT1M data. This reproduces Table 2 in the original paper [Matsui+, ACMMM 18].

First, please download the data by the following script. They will be stored in `./data`.
The whole data require 745 MB disk space.
```bash
bash download_sift1m.sh
```

Then let's install ANN methods we'd like to compare. The anaconda environment is required for testing faiss:
```bash
pip install annoy FALCONN rii nmslib
conda install -c pytorch faiss-cpu
```

Now you can run the benchmark script:
```bash
python sift1m.py
```



## SIFT1B

Here, let's evaluate the runtime/accuracy using SIFT1B data.

Please download the SIFT1B data by the following script, that will be stored in `./data`.
This will take several hours. The whole data require 255 GB disk space.
```bash
bash download_sift1b.sh
```

Please install a helper libraries to read a large bvecs file with an iterator
```bash
pip install texmex-python more-itertools
```

Now you can run the benchmark script:
```bash
python sift1b.py
```

Note that the codec and the search instane will be cached in `./cache`. You can use files there to run the search again.
