// C ABI over tt-metal's own host BFP tile packers (tt_metal/impl/data_format), for the bit-exactness test of
// fnx.quant.bfp (ttnn has no bfloat2_b DataType, so Bfp2_b cannot be reached through ttnn.from_torch).
// Input: tile-contiguous float32 (tile k = values [1024k, 1024k+1024), each tile row-major 32x32), exp_b format.
// Built inside the TT image by fnx/tests/test_quant_bfp.py.
#include <cstdint>
#include <cstring>
#include <vector>

#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt_stl/span.hpp>

extern "C" {

// fmt: 2, 4 or 8. Returns the number of uint32 words written (<= cap), or -1.
int64_t ttbfp_pack(int fmt, const float* data, int64_t n, uint32_t* out, int64_t cap) {
    ttsl::Span<const float> s(data, static_cast<size_t>(n));
    std::vector<uint32_t> w;
    if (fmt == 2) {
        w = pack_as_bfp2_tiles<float>(s, /*row_major_input=*/true, /*is_exp_a=*/false);
    } else if (fmt == 4) {
        w = pack_as_bfp4_tiles<float>(s, true, false);
    } else if (fmt == 8) {
        w = pack_as_bfp8_tiles<float>(s, true, false);
    } else {
        return -1;
    }
    if (static_cast<int64_t>(w.size()) > cap) return -1;
    std::memcpy(out, w.data(), w.size() * sizeof(uint32_t));
    return static_cast<int64_t>(w.size());
}

// Inverse: words -> n floats (tile-contiguous, row-major inside each tile). Returns n or -1.
int64_t ttbfp_unpack(int fmt, const uint32_t* words, int64_t nw, float* out, int64_t n) {
    ttsl::Span<const uint32_t> s(words, static_cast<size_t>(nw));
    std::vector<float> v;
    if (fmt == 2) {
        v = unpack_bfp2_tiles_into_float_vec(s, /*row_major_output=*/true, /*is_exp_a=*/false);
    } else if (fmt == 4) {
        v = unpack_bfp4_tiles_into_float_vec(s, true, false);
    } else if (fmt == 8) {
        v = unpack_bfp8_tiles_into_float_vec(s, true, false);
    } else {
        return -1;
    }
    if (static_cast<int64_t>(v.size()) != n) return -1;
    std::memcpy(out, v.data(), v.size() * sizeof(float));
    return n;
}

}  // extern "C"
