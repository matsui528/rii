#ifndef RII_H
#define RII_H

#include <iostream>
#include <cassert>
#include "./pqkmeans.h"
#include "./distance.h"

// Handle missing ssize_t on Windows. 
# if defined(_MSC_VER) 
    typedef __int64 ssize_t;
# endif

// For py::array_t
// See http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html#direct-access
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


namespace rii {


struct DistanceTable{
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    DistanceTable() {}
    DistanceTable(size_t M, size_t Ks) : Ks_(Ks), data_(M * Ks) {}
    void SetVal(size_t m, size_t ks, float val) {
        data_[m * Ks_ + ks] = val;
    }
    float GetVal(size_t m, size_t ks) const {
        return data_[m * Ks_ + ks];
    }
    size_t Ks_;
    std::vector<float> data_;
};



class RiiCpp {
public:
    RiiCpp() {}  // Shouldn't be default-constructed
    RiiCpp(const py::array_t<float> &codewords, bool verbose);

    // ===== Functions that can be called from Python =====
    //void SetCodewords(const py::array_t<float> &codewords);  // This should be called first
    void Reconfigure(int nlist, int iter);
    void AddCodes(const py::array_t<unsigned char> &codes, bool update_flag);

    // The default integers of Python is int64 (long long), so the type of target_ids is long long
    std::pair<std::vector<size_t>, std::vector<float>> QueryLinear(const py::array_t<float> &query,
                                                                int topk,
                                                                const py::array_t<long long> &target_ids) const;
    std::pair<std::vector<size_t>, std::vector<float>> QueryIvf(const py::array_t<float> &query,
                                                             int topk,
                                                             const py::array_t<long long> &target_ids,
                                                             int L) const;
    void Clear();

    // ===== Functions that would not be called from Python (Used inside c++) =====
    void UpdatePostingLists(size_t start, size_t num);
    DistanceTable DTable(const py::array_t<float> &vec) const;
    float ADist(const DistanceTable &dtable, const std::vector<unsigned char> &code) const;
    float ADist(const DistanceTable &dtable, const std::vector<unsigned char> &flattened_codes, size_t n) const;
    std::pair<std::vector<size_t>, std::vector<float>> PairVectorToVectorPair(const std::vector<std::pair<size_t, float>> &pair_vec) const;

    // Property getter
    size_t GetN() const {return flattened_codes_.size() / M_;}
    size_t GetNumList() const {return coarse_centers_.size();}

    // Given a long (N * M) codes, pick up n-th code
    std::vector<unsigned char> NthCode(const std::vector<unsigned char> &long_code, size_t n) const;
    // Given a long (N * M) codes, pick up m-th element from n-th code
    unsigned char NthCodeMthElement(const std::vector<unsigned char> &long_code, std::size_t n, size_t m) const;

    // Member variables
    size_t M_, Ks_;
    bool verbose_;
    std::vector<std::vector<std::vector<float>>> codewords_;  // (M, Ks, Ds)
    std::vector<std::vector<unsigned char>> coarse_centers_;  // (NumList, M)
    std::vector<unsigned char> flattened_codes_;  // (N, M) PQ codes are flattened to N * M long array
    std::vector<std::vector<int>> posting_lists_;  // (NumList, any)
};


RiiCpp::RiiCpp(const py::array_t<float> &codewords, bool verbose)
{
    verbose_ = verbose;
    const auto &r = codewords.unchecked<3>();  // codewords must have ndim=3, with non-writable
    M_ = (size_t) r.shape(0);
    Ks_ = (size_t) r.shape(1);
    size_t Ds = (size_t) r.shape(2);
    codewords_.resize(M_, std::vector<std::vector<float>>(Ks_, std::vector<float>(Ds)));
    for (ssize_t m = 0; m < r.shape(0); ++m) {
        for (ssize_t ks = 0; ks < r.shape(1); ++ks) {
            for (ssize_t ds = 0; ds < r.shape(2); ++ds) {
                codewords_[m][ks][ds] = r(m, ks, ds);
            }
        }
    }

    if (verbose_) {
        // Check which SIMD functions are used. See distance.h for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

void RiiCpp::Reconfigure(int nlist, int iter)
{
    assert(0 < nlist);
    assert((size_t) nlist <= GetN());

    // ===== (1) Sampling vectors for pqk-means =====
    // Since clustering takes time, we use a subset of all codes for clustering.
    size_t len_for_clustering = std::min(GetN(), (size_t) nlist * 100);
    if (verbose_) {
        std::cout << "The number of vectors used for training of coarse centers: " << len_for_clustering << std::endl;
    }
    // Prepare a random set of integers, drawn from [0, ..., N-1], where the cardinality of the set is len_for_clustering
    std::vector<size_t> ids_for_clustering(GetN());  // This can be large and might be the bootle neck of memory consumption
    std::iota(ids_for_clustering.begin(), ids_for_clustering.end(), 0);  // 0, 1, 2, ...
    std::shuffle(ids_for_clustering.begin(), ids_for_clustering.end(), std::default_random_engine(123));
    ids_for_clustering.resize(len_for_clustering);
    ids_for_clustering.shrink_to_fit();  // For efficient memory usage

    std::vector<unsigned char> flattened_codes_randomly_picked;  // size=len_for_clustering
    flattened_codes_randomly_picked.reserve(len_for_clustering * M_);
    for (const auto &id : ids_for_clustering) {  // Pick up vectors to construct a training set
        std::vector<unsigned char> code = NthCode(flattened_codes_, id);
        flattened_codes_randomly_picked.insert(flattened_codes_randomly_picked.end(),
                                               code.begin(), code.end());
    }
    assert(flattened_codes_randomly_picked.size() == len_for_clustering * M_);


    // ===== (2) Run pqk-means =====
    if (verbose_) {std::cout << "Start to run PQk-means" << std::endl;}
    pqkmeans::PQKMeans clustering_instance(codewords_, nlist, iter, verbose_);
    clustering_instance.fit(flattened_codes_randomly_picked);


    // ===== (3) Update coarse centers =====
    coarse_centers_ = clustering_instance.GetClusterCenters();
    assert(coarse_centers_.size() == (size_t) nlist);
    assert(coarse_centers_[0].size() == M_);


    // ===== (4) Update posting lists =====
    if (verbose_) {std::cout << "Start to update posting lists" << std::endl;}
    posting_lists_.clear();
    posting_lists_.resize(nlist);
    for (auto &posting_list : posting_lists_) {
        posting_list.reserve(GetN() / nlist);  // Roughly malloc
    }
    UpdatePostingLists(0, GetN());
}

void RiiCpp::AddCodes(const py::array_t<unsigned char> &codes, bool update_flag)
{
    // (1) Add new input codes to flatted_codes. This imply pushes back the elements.
    // After that, if update_flg=true, (2) update posting lists for the input codes.
    // Note that update_flag should be true in usual cases. It should be false
    // if (1) this is the first call of AddCodes (i.e., calling in add_configure()),
    // of (2) you've decided to call reconfigure() manually after add()

    if (update_flag && coarse_centers_.empty()) {
        std::cerr << "Error. reconfigure() must be called before running add(vecs=X, update_posting_lists=True)."
                  << "If this is the first addition, please call add_configure(vecs=X)" << std::endl;
        throw;
    }

    // ===== (1) Add codes to flattened_codes =====
    const auto &r = codes.unchecked<2>(); // codes must have ndim=2; with non-writeable
    size_t N = (size_t) r.shape(0);
    assert(M_ == (size_t) r.shape(1));
    size_t N0 = GetN();
    flattened_codes_.resize( (N0 + N) * M_);
    for (size_t n = 0; n < N; ++n) {
        for (size_t m = 0; m < M_; ++m) {
            flattened_codes_[ (N0 + n) * M_ + m] = r(n, m);
        }
    }
    if (verbose_) {
        std::cout << N << " new vectors are added." << std::endl;
        std::cout << "Total number of codes is " << GetN() << std::endl;
    }

    // ===== (2) Update posting lists =====
    if (update_flag) {
        if (verbose_) { std::cout << "Start to update posting lists" << std::endl; }
        UpdatePostingLists(N0, N);
    }
}

std::pair<std::vector<size_t>, std::vector<float> > RiiCpp::QueryLinear(const py::array_t<float> &query,
                                                                        int topk,
                                                                        const py::array_t<long long> &target_ids) const
{
    const auto &tids = target_ids.unchecked<1>(); // target_ids must have ndim = 1; can be non-writeable
    size_t S = tids.shape(0);  // The number of target_ids. It might be 0 if not specified.

    assert((size_t) topk <= GetN());

    // ===== (1) Create dtable =====
    DistanceTable dtable = DTable(query);

    // ===== (2) Run PQ linear search =====
    // [todo] Can be SIMDized?
    std::vector<std::pair<size_t, float>> scores;
    if (S == 0) {  // No target ids
        size_t N = GetN();
        scores.resize(N);
#pragma omp parallel for
        for (long long n_tmp = 0LL; n_tmp < static_cast<long long>(N); ++n_tmp) {
            size_t n = static_cast<size_t>(n_tmp);
            scores[n] = {n, ADist(dtable, flattened_codes_, n)};
        }
    } else {  // Target ids are specified
        assert((size_t) topk <= S);
        assert(S <= GetN());
        scores.resize(S);
#pragma omp parallel for
        for (long long s_tmp = 0LL; s_tmp < static_cast<long long>(S); ++s_tmp) {
            size_t s = static_cast<size_t>(s_tmp);
            size_t tid = static_cast<size_t>(tids(s));
            scores[s] = {tid, ADist(dtable, flattened_codes_, tid)};
        }
    }



    // ===== (3) Sort them =====
    // [todo] Can be parallelized?
    std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                      [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
    scores.resize(topk);
    scores.shrink_to_fit();

    // ===== (4) Return the result, in the form of pair<vec, vec> =====
    // Note that this returns two lists, not np.array
    return PairVectorToVectorPair(scores);
}

std::pair<std::vector<size_t>, std::vector<float> > RiiCpp::QueryIvf(const py::array_t<float> &query,
                                                                     int topk,
                                                                     const py::array_t<long long> &target_ids,
                                                                     int L) const
{
    const auto &tids = target_ids.unchecked<1>(); // target_ids must have ndim = 1 with non-writeable
    size_t S = tids.shape(0);  // The number of target_ids. It might be 0 if not specified.

    assert((size_t) topk <= GetN());
    assert(topk <= L && (size_t) L <= GetN());

    // ===== (1) Create dtable =====
    DistanceTable dtable = DTable(query);

    // ===== (2) Compare to coarse centers and sort the results =====
    std::vector<std::pair<size_t, float>> scores_coarse(coarse_centers_.size());
    size_t nlist = GetNumList();
//#pragma omp parallel for
    for (size_t no = 0; no < nlist; ++no) {
        scores_coarse[no] = {no, ADist(dtable, coarse_centers_[no])};
    }

    // ===== (3) Partial sort the coarse results. =====
    size_t w;  // The number of posting lists to be considered
    if (S == 0) {
        w = (size_t) std::round((double) L * GetNumList() / GetN());
    } else {
        assert((size_t) topk <= S && S <= GetN());
        w = (size_t) std::round((double) L * GetNumList() / S);
    }
    w += 3;  // Top poslists might contain a few items, so we set w litter bit bigger for insurance
    if (nlist < w) {  // If w is bigger than the original nlist, let's set back nlist
        w = nlist;
    }

    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + w, scores_coarse.end(),
                      [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});

    // ===== (4) Traverse posting list =====
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    int coarse_cnt = 0;
    for (const auto &score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        coarse_cnt++;

        // [todo] This loop can be parallelized
        for (const auto &n : posting_lists_[no]) {
            // ===== (5) If id is not included in target_ids, skip. =====
            // Note that if S==0 (target is all), then evaluate all IDs
            if (S != 0 && !std::binary_search(target_ids.data(), target_ids.data() + S, static_cast<long long>(n))) {
                continue;
            }

            // ===== (6) Evaluate n =====
            scores.emplace_back(n, ADist(dtable, flattened_codes_, n));

            // ===== (7) If scores are collected enough =====
            if (scores.size() == (size_t) L) {
                goto finish;
            }
        }

        // If w coarse centers are traversed and still L items are not found while more than topk items are found,
        // we terminate the process and do the final reranking
        if ( (size_t) coarse_cnt == w && scores.size() >= (unsigned long) topk) {
finish:
            // ===== (8) Sort them =====
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                              [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
            scores.resize(topk);
            scores.shrink_to_fit();

            // ===== (9) Return the result, in the form of pair<vec, vec> =====
            // Note that this returns two lists, not np.array
            return PairVectorToVectorPair(scores);
        }

    }

    // It can be happened that vectors are not found
    return std::pair<std::vector<size_t>, std::vector<float>>({}, {});
}

void RiiCpp::Clear()
{
    coarse_centers_.clear();
    flattened_codes_.clear();
    posting_lists_.clear();
}

void RiiCpp::UpdatePostingLists(size_t start, size_t num)
{
    // Update (add) identifiers to posting lists, from codes[start] to codes[start + num -1]
    // This just add IDs, so be careful to call this (e.g., the same IDs will be added if you call
    // this funcs twice at the same time, that would be not expected behavior)
    assert(start <= GetN());
    assert(start + num <= GetN());


    // ===== (1) Construct a dummy pqkmeans class for computing Symmetric Distance =====
    pqkmeans::PQKMeans clustering_instance(codewords_, (int)GetNumList(), 0, true);
    clustering_instance.SetClusterCenters(coarse_centers_);

    // ===== (2) Update posting lists =====
    std::vector<size_t> assign(num);
#pragma omp parallel for
    for (long long n_tmp = 0LL; n_tmp < static_cast<long long>(num); ++n_tmp) {
        size_t n = static_cast<size_t>(n_tmp);
        assign[n] = clustering_instance.predict_one(NthCode(flattened_codes_, start + n));
    }

    for (size_t n = 0; n < num; ++n) {
        posting_lists_[assign[n]].push_back((int)(start + n));
    }
}

DistanceTable RiiCpp::DTable(const py::array_t<float> &vec) const
{
    const auto &v = vec.unchecked<1>();
    size_t Ds = codewords_[0][0].size();
    assert((size_t) v.shape(0) == M_ * Ds);
    DistanceTable dtable(M_, Ks_);
    for (size_t m = 0; m < M_; ++m) {
        for (size_t ks = 0; ks < Ks_; ++ks) {
            dtable.SetVal(m, ks, fvec_L2sqr(&(v(m * Ds)), codewords_[m][ks].data(), Ds));
        }
    }
    return dtable;
}

float RiiCpp::ADist(const DistanceTable &dtable, const std::vector<unsigned char> &code) const
{
    assert(code.size() == M_);
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        unsigned char ks = code[m];
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

float RiiCpp::ADist(const DistanceTable &dtable, const std::vector<unsigned char> &flattened_codes, size_t n) const
{
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        unsigned char ks = NthCodeMthElement(flattened_codes, n, m);
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

std::pair<std::vector<size_t>, std::vector<float> > RiiCpp::PairVectorToVectorPair(const std::vector<std::pair<size_t, float> > &pair_vec) const
{
    std::pair<std::vector<size_t>, std::vector<float>> vec_pair(std::vector<size_t>(pair_vec.size()), std::vector<float>(pair_vec.size()));
    for(size_t n = 0, N = pair_vec.size(); n < N; ++n) {
        vec_pair.first[n] = pair_vec[n].first;
        vec_pair.second[n] = pair_vec[n].second;
    }
    return vec_pair;
}



std::vector<unsigned char> RiiCpp::NthCode(const std::vector<unsigned char> &long_code, size_t n) const
{
    return std::vector<unsigned char>(long_code.begin() + n * M_, long_code.begin() + (n + 1) * M_);
}

unsigned char RiiCpp::NthCodeMthElement(const std::vector<unsigned char> &long_code, std::size_t n, size_t m) const
{
    return long_code[ n * M_ + m];
}


} // namespace rii

#endif // RII_H
