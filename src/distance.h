#ifndef DISTANCE_H
#define DISTANCE_H

// http://koturn.hatenablog.com/entry/2016/07/18/090000
// windows is not supported, but just in case (later someone might implement)
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=590,27,2
#ifdef _MSC_VER
#  include <immintrin.h> 
#else
#  include <x86intrin.h>
#endif

// These fast L2 squared distance codes (SSE and AVX) are from the Faiss library:
// https://github.com/facebookresearch/faiss/blob/master/utils.cpp
//
// Based on them, AVX512 implementation is also prepared.
// But it doesn't seem drastically fast. Only slightly faster than AVX:
// (runtime) REF >> SSE >= AVX ~ AVX512

namespace rii {

// From Faiss.
// Reference implementation
float fvec_L2sqr_ref(const float *x, const float *y, size_t d)
{
    size_t i;
    float res_ = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res_ += tmp * tmp;
    }
    return res_;
}


// ========================= Reading functions ============================

// Reading function for SSE, AVX, and AVX512
// This function is from Faiss
// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);

#if defined(_MSC_VER)
    //alignas (alignof(16)) float buf[4] = { 0, 0, 0, 0 };
    __declspec(align(16)) float buf[4] = { 0, 0, 0, 0 };
#else
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
#endif

    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

#if defined(__AVX__)
// Reading function for AVX and AVX512
// This function is from Faiss
// reads 0 <= d < 8 floats as __m256
static inline __m256 masked_read_8 (int d, const float *x)
{
    assert (0 <= d && d < 8);
    if (d < 4) {
        __m256 res = _mm256_setzero_ps ();
        res = _mm256_insertf128_ps (res, masked_read (d, x), 0);
        return res;
    } else {
        __m256 res = _mm256_setzero_ps ();
        res = _mm256_insertf128_ps (res, _mm_loadu_ps (x), 0);
        res = _mm256_insertf128_ps (res, masked_read (d - 4, x + 4), 1);
        return res;
    }
}
#endif // __AVX__



 #if defined(__AVX512F__) 
// Reading function for AVX512
// reads 0 <= d < 16 floats as __m512
static inline __m512 masked_read_16 (int d, const float *x)
{
    assert (0 <= d && d < 16);
    if (d < 8) {
        __m512 res = _mm512_setzero_ps ();
        res = _mm512_insertf32x8 (res, masked_read_8 (d, x), 0);
        return res;
    } else {
        __m512 res = _mm512_setzero_ps ();
        res = _mm512_insertf32x8 (res, _mm256_loadu_ps (x), 0);
        res = _mm512_insertf32x8 (res, masked_read_8 (d - 8, x + 8), 1);
        return res;
    }
}

#endif // __AVX512F__



// ========================= Distance functions ============================

#if defined(__AVX512F__)  
static const std::string g_simd_architecture = "avx512";

// AVX512 implementation by Yusuke
float fvec_L2sqr (const float *x, const float *y, size_t d)
{
    __m512 msum1 = _mm512_setzero_ps();

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        const __m512 a_m_b1 = _mm512_sub_ps(mx, my);
        //msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm512_add_ps(msum1, _mm512_mul_ps(a_m_b1, a_m_b1));
        d -= 16;
    }

    __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
    // msum2 += _mm512_extractf32x8_ps(msum1, 0);
    msum2 = _mm256_add_ps(msum2, _mm512_extractf32x8_ps(msum1, 0));

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        // const __m256 a_m_b1 = mx - my;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm256_add_ps(msum2, _mm256_mul_ps(a_m_b1, a_m_b1));
        d -= 8;
    }

    __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
    // msum3 += _mm256_extractf128_ps(msum2, 0);
    msum3 = _mm_add_ps(msum3, _mm256_extractf128_ps(msum2, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum3 += a_m_b1 * a_m_b1;
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum3 += a_m_b1 * a_m_b1;
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum3 = _mm_hadd_ps (msum3, msum3);
    msum3 = _mm_hadd_ps (msum3, msum3);
    return  _mm_cvtss_f32 (msum3);
}

#elif defined (__AVX__)  
static const std::string g_simd_architecture = "avx";

// This function is from Faiss
// AVX implementation
float fvec_L2sqr (const float *x, const float *y, size_t d)
{
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        // const __m256 a_m_b1 = mx - my;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1 ,a_m_b1));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    // msum2 += _mm256_extractf128_ps(msum1, 0);
    msum2 =  _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

#else 
static const std::string g_simd_architecture = "sse";


// This function is from Faiss
// SSE implementation
float fvec_L2sqr (const float *x, const float *y, size_t d)
{
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum1 = _mm_hadd_ps (msum1, msum1);
    msum1 = _mm_hadd_ps (msum1, msum1);
    return  _mm_cvtss_f32 (msum1);
}


#endif




} // namespace rii

#endif // DISTANCE_H
