#include <stddef.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#if defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE float dot_scalar(const float *a, const float *b, size_t count) {
    float total = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < count; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

NOINLINE float dot_vector(const float *a, const float *b, size_t count) {
    size_t i = 0;
    float total = 0.0f;
#if defined(__aarch64__)
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    for (; i + 4 <= count; i += 4) {
        accumulator = vmlaq_f32(accumulator, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    total = vaddvq_f32(accumulator);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 accumulator = _mm256_setzero_ps();
    for (; i + 8 <= count; i += 8) {
        accumulator = _mm256_fmadd_ps(
            _mm256_loadu_ps(a + i),
            _mm256_loadu_ps(b + i),
            accumulator
        );
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, accumulator);
    for (size_t lane = 0; lane < 8; ++lane) total += lanes[lane];
#endif
    for (; i < count; ++i) total += a[i] * b[i];
    return total;
}

const char *vector_backend(void) {
#if defined(__aarch64__)
    return "ARM NEON (128-bit, 4 x float32 lanes)";
#elif defined(__AVX2__) && defined(__FMA__)
    return "x86 AVX2+FMA (256-bit, 8 x float32 lanes)";
#else
    return "portable scalar fallback";
#endif
}

NOINLINE double dot_reference(const float *a, const float *b, size_t count) {
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        total += (double)a[i] * (double)b[i];
    }
    return total;
}
