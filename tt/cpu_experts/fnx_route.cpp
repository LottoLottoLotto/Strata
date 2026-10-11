// fnx_route.cpp - the cpu MoE's host routing step (fnx.tt.moe.Moe.plan) in one native call: top-10 of the device
// router logits with exact refinement of near ties, the route weights, the CPU / device slot split and, for a call of
// at most 32 rows (one 32-row group), the device experts' metadata row (fnx.tt.moe.DeviceExperts.meta_words).
//
// Semantics (fnx/tt/moe.py module docstring):
//   order    device logits descending, ties -> the lower id (numpy's stable argsort of -lg; -0 == +0)
//   bound    b_e = (PRODUCT_ERR * sqrt(sum_d f64(x_d * x_d))) * |W_e| (f64); lo = lg - b, hi = lg + b
//   sure     lo_e > the 11th largest hi: e is surely among the top 10 (at most 10 such experts)
//   refine   rows with fewer than 10 sure experts: the sure ones in device order, then the undecided experts
//            (hi >= the 10th largest lo) by their exact logit f32(sum_d f64(W_e,d) * f64(x_d)) descending, ties ->
//            the lower id
//   weights  softmax over the 10 selected device logits in f64 (numpy's pairwise sum of the 10 exponentials), f32
// The f64 sums run sequentially here (numpy: pairwise / BLAS): the bound and the exact logits may differ from the
// numpy version in their last f64 bits, which cannot change a pick unless a value sits within ~1e-16 of a tie.
// Every caller (eager and traced) uses this one routine, so their routings are identical.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr int K = 10;
constexpr int HDR = 64;     // fnx.tt.moe.HDR

// numpy's pairwise summation of n doubles (umath loops_utils pairwise_sum, PW_BLOCKSIZE 128)
double pairwise(const double* a, int n) {
    if (n < 8) {
        double r = 0.0;
        for (int i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

}  // namespace

extern "C" {

/// Route T rows.  lg f32 [T, E] device router logits, x f32 [T, D] the same rows, wr f32 [E, D] the router weights
/// (their BF16 values), wr_norm f64 [E] = |W_e| (f64), local int32 [E] the device index of a resident expert (-1:
/// CPU).  Writes ids int32 [T, 10] (device-logit rank order, exact order among refined picks), wt f32 [T, 10],
/// cpu_ids int32 [T, 10] (ids, -1 where the expert is resident).  words (non-null, T <= 32): the metadata row
/// uint32 [HDR + 33 * s_cap] of the resident slots (meta_words of DeviceExperts.entries: one group; a call without
/// resident slots gets local expert 0 at weight 0), *n_entries its entry count.  Returns the number of refined rows,
/// -1 when the entries exceed s_cap, -2 for T > 32 with words.
int fnx_route(int T, int E, int D, const float* lg, const float* x, const float* wr, const double* wr_norm,
              double product_err, const int32_t* local, int32_t* ids, float* wt, int32_t* cpu_ids, uint32_t* words,
              int s_cap, int* n_entries) {
    if (words && T > 32) return -2;
    std::vector<int> idx(E);
    std::vector<double> lo(E), hi(E), tmp(E);
    std::vector<uint8_t> sure(E);
    int refined = 0;
    auto before = [](const float* l) {        // device order: logit descending, then id ascending
        return [l](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); };
    };
    for (int t = 0; t < T; ++t) {
        const float* l = lg + (size_t) t * E;
        const float* xt = x + (size_t) t * D;
        int32_t* out = ids + (size_t) t * K;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), before(l));
        double ss = 0.0;
        for (int d = 0; d < D; ++d) {
            const float p = xt[d] * xt[d];
            ss += (double) p;
        }
        const double s = product_err * std::sqrt(ss);
        for (int e = 0; e < E; ++e) {
            const double b = s * wr_norm[e];
            lo[e] = (double) l[e] - b;
            hi[e] = (double) l[e] + b;
        }
        tmp = hi;
        std::nth_element(tmp.begin(), tmp.begin() + (E - K - 1), tmp.end());
        const double ups = tmp[E - K - 1];                          // the 11th largest upper bound
        int n_in = 0;
        for (int e = 0; e < E; ++e) n_in += (sure[e] = lo[e] > ups);
        if (n_in >= K) {
            for (int k = 0; k < K; ++k) out[k] = idx[k];
        } else {
            ++refined;
            tmp = lo;
            std::nth_element(tmp.begin(), tmp.begin() + (E - K), tmp.end());
            const double tau = tmp[E - K];                          // the 10th largest lower bound
            std::vector<int> ins, amb;
            for (int e = 0; e < E; ++e) {
                if (sure[e]) ins.push_back(e);
                else if (hi[e] >= tau) amb.push_back(e);
            }
            std::sort(ins.begin(), ins.end(), before(l));
            std::vector<float> ex(amb.size());
            for (size_t i = 0; i < amb.size(); ++i) {
                const float* w = wr + (size_t) amb[i] * D;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;     // 4 lanes (vectorized); D % 4 == 0
                for (int d = 0; d < D; d += 4) {
                    a0 += (double) w[d] * (double) xt[d];
                    a1 += (double) w[d + 1] * (double) xt[d + 1];
                    a2 += (double) w[d + 2] * (double) xt[d + 2];
                    a3 += (double) w[d + 3] * (double) xt[d + 3];
                }
                ex[i] = (float) ((a0 + a1) + (a2 + a3));
            }
            std::vector<int> ord(amb.size());
            std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                return ex[a] > ex[b] || (ex[a] == ex[b] && amb[a] < amb[b]);
            });
            int k = 0;
            for (int e : ins) out[k++] = e;
            for (size_t i = 0; k < K && i < ord.size(); ++i) out[k++] = amb[ord[i]];
        }
        double sl[K], m = -INFINITY;
        for (int k = 0; k < K; ++k) m = std::max(m, sl[k] = (double) l[out[k]]);
        double e[K];
        for (int k = 0; k < K; ++k) e[k] = std::exp(sl[k] - m);
        const double sum = pairwise(e, K);
        for (int k = 0; k < K; ++k) {
            wt[(size_t) t * K + k] = (float) (e[k] / sum);
            cpu_ids[(size_t) t * K + k] = local[out[k]] >= 0 ? -1 : out[k];
        }
    }
    if (words) {
        // entries of the one group: resident local ids ascending; route[s][row] = that slot's weight
        std::vector<int> loc_of;
        for (int i = 0; i < T * K; ++i)
            if (local[ids[i]] >= 0) loc_of.push_back(local[ids[i]]);
        std::sort(loc_of.begin(), loc_of.end());
        loc_of.erase(std::unique(loc_of.begin(), loc_of.end()), loc_of.end());
        const bool none = loc_of.empty();
        if (none) loc_of.push_back(0);
        const int S = (int) loc_of.size();
        if (S > s_cap) return -1;
        std::memset(words, 0, sizeof(uint32_t) * (size_t) (HDR + 33 * s_cap));
        words[0] = (uint32_t) S;
        words[1] = 1;
        words[2] = 0;
        words[3] = (uint32_t) T;
        words[4] = 0;
        words[5] = (uint32_t) S;
        for (int s = 0; s < S; ++s) words[HDR + s] = (uint32_t) loc_of[s];
        float* route = reinterpret_cast<float*>(words + HDR + s_cap);
        if (!none) {
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < K; ++k) {
                    const int lc = local[ids[t * K + k]];
                    if (lc < 0) continue;
                    const int s = (int) (std::lower_bound(loc_of.begin(), loc_of.end(), lc) - loc_of.begin());
                    route[(size_t) s * 32 + t] = wt[(size_t) t * K + k];
                }
        }
        *n_entries = S;
    }
    return refined;
}

}  // extern "C"
