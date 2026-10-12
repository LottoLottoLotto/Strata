// fnx_qsa.cpp - the per-row assembly of the QSA sparse selection (fnx.tt.qsa.Qsa._select): from the bounds' verdicts
// (surely in / candidate, for the top K blocks and the top K + 1) and the exact scores of the candidates, the row's
// ascending cell list.  Same rules as the numpy code it replaces:
//   top(k)  the blocks surely in (ascending), then the k - |sure| best candidates by exact score descending, ties ->
//           the lower block (np.lexsort((c, -ex)));
//   cells   the 4 cells of each block of top(K), the first e cells of the one block in top(K + 1) but not top(K)
//           (e = 3 - n % 4 > 0), and the open tail nbt * 4 .. n - 1 (forced by the bias), ascending: SEL_WIDTH cells.
// The exact scores come from the caller (fnx.ref.qsa.block_scores: the reference's own arithmetic).
#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" {

/// R rows over nbm blocks.  sure, cand: uint8 [2][R][nbm] (index 0: top K, 1: top K + 1); ex: f32 [R][nbm], the exact
/// score of every candidate block (other entries unread); e, nbt, n: int32 [R]; K the block count (512); width the
/// cells per row (SEL_WIDTH).  Writes cells int32 [R][width].  Returns 0, or -(1 + row) when a row's candidates are
/// fewer than its open slots or its cell count is not `width` (the bounds were violated).
int fnx_qsa_assemble(int R, int nbm, int K, const uint8_t* sure, const uint8_t* cand, const float* ex,
                     const int32_t* e, const int32_t* nbt, const int32_t* n, int width, int32_t* cells) {
    std::vector<int> top_k, top_k1, c;
    std::vector<uint8_t> in(nbm);
    auto top = [&](int which, int i, int k, std::vector<int>& out) -> bool {
        const uint8_t* s = sure + ((size_t) which * R + i) * nbm;
        const uint8_t* cd = cand + ((size_t) which * R + i) * nbm;
        const float* x = ex + (size_t) i * nbm;
        out.clear();
        c.clear();
        for (int b = 0; b < nbm; ++b) {
            if (s[b]) out.push_back(b);
            else if (cd[b]) c.push_back(b);
        }
        const int r = k - (int) out.size();
        if (r <= 0) return r == 0;
        if ((int) c.size() < r) return false;
        std::partial_sort(c.begin(), c.begin() + r, c.end(),
                          [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
        out.insert(out.end(), c.begin(), c.begin() + r);
        return true;
    };
    for (int i = 0; i < R; ++i) {
        if (!top(0, i, K, top_k)) return -(1 + i);
        int32_t* out = cells + (size_t) i * width;
        int m = 0;
        std::fill(in.begin(), in.end(), 0);
        for (int b : top_k) in[b] = 1;
        int extra = -1;
        if (e[i] > 0) {
            if (!top(1, i, K + 1, top_k1)) return -(1 + i);
            for (int b : top_k1)
                if (!in[b]) {
                    if (extra >= 0) return -(1 + i);
                    extra = b;
                }
            if (extra < 0) return -(1 + i);
        }
        // ascending: blocks in order, the extra block's first e cells in its place, then the tail
        for (int b = 0; b < nbm && m <= width; ++b) {
            if (in[b])
                for (int j = 0; j < 4; ++j) out[m++] = 4 * b + j;
            else if (b == extra)
                for (int j = 0; j < e[i]; ++j) out[m++] = 4 * b + j;
            if (m > width) return -(1 + i);
        }
        for (int cpos = nbt[i] * 4; cpos < n[i]; ++cpos) {
            if (m >= width) return -(1 + i);
            out[m++] = cpos;
        }
        if (m != width) return -(1 + i);
    }
    return 0;
}

}  // extern "C"
