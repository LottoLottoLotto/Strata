// Top-10 routing of one token row on a RISC-V data-movement core (integer arithmetic only: no FPU) for the fused
// router (router_writer.cpp).  route_row(v, out): v = the row's N_EXP + 1 f32 logits (bits; column N_EXP = shared gate),
// out = its 32-word route row (common.h): ids in descending logit order (ties -> the lower id: insertion keeps the
// earlier id first among equal keys), softmax over the 10 selected logits (logits -> signed Q20, exp(-d) in Q31 from
// a 2^(-i/128) table + 2nd-order term, rel. error < 3e-8; normalisation through one Q62 reciprocal, rounded to f32:
// ~1e-7 overall) and sigmoid(shared gate) the same way.
#pragma once
#include <stdint.h>

#include "common.h"

namespace fnx_moe {
constexpr uint32_t POW2_NEG128[129] = {
    2147483648u, 2135885998u, 2124350982u, 2112878262u, 2101467502u, 2090118366u, 2078830522u, 2067603638u,
    2056437387u, 2045331439u, 2034285470u, 2023299156u, 2012372174u, 2001504204u, 1990694927u, 1979944027u,
    1969251188u, 1958616096u, 1948038440u, 1937517909u, 1927054196u, 1916646992u, 1906295993u, 1896000896u,
    1885761398u, 1875577199u, 1865448001u, 1855373507u, 1845353420u, 1835387448u, 1825475297u, 1815616678u,
    1805811301u, 1796058879u, 1786359126u, 1776711757u, 1767116489u, 1757573041u, 1748081133u, 1738640488u,
    1729250827u, 1719911875u, 1710623359u, 1701385007u, 1692196547u, 1683057710u, 1673968228u, 1664927835u,
    1655936265u, 1646993254u, 1638098541u, 1629251865u, 1620452965u, 1611701585u, 1602997467u, 1594340357u,
    1585730000u, 1577166143u, 1568648537u, 1560176931u, 1551751076u, 1543370725u, 1535035634u, 1526745556u,
    1518500250u, 1510299473u, 1502142985u, 1494030547u, 1485961921u, 1477936870u, 1469955159u, 1462016553u,
    1454120821u, 1446267730u, 1438457051u, 1430688553u, 1422962010u, 1415277195u, 1407633882u, 1400031848u,
    1392470869u, 1384950723u, 1377471191u, 1370032052u, 1362633090u, 1355274085u, 1347954824u, 1340675091u,
    1333434672u, 1326233356u, 1319070932u, 1311947188u, 1304861917u, 1297814910u, 1290805962u, 1283834865u,
    1276901417u, 1270005413u, 1263146652u, 1256324931u, 1249540052u, 1242791816u, 1236080024u, 1229404479u,
    1222764986u, 1216161350u, 1209593378u, 1203060876u, 1196563654u, 1190101520u, 1183674286u, 1177281762u,
    1170923762u, 1164600099u, 1158310587u, 1152055042u, 1145833280u, 1139645120u, 1133490379u, 1127368878u,
    1121280436u, 1115224875u, 1109202018u, 1103211687u, 1097253708u, 1091327906u, 1085434106u, 1079572136u,
    1073741824u};

// Monotonic unsigned key of an f32 bit pattern (larger float -> larger key).
inline uint32_t order_key(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

// f32 bits -> signed Q20 (|x| clamped below 2^11; router logits are far smaller).
inline int32_t f32_to_q20(uint32_t b) {
    const int32_t e = (b >> 23) & 0xff;
    if (e == 0) {
        return 0;
    }
    const uint32_t m = (b & 0x7fffffu) | 0x800000u;     // value = m * 2^(e - 150), Q20 = m * 2^(e - 130)
    const int32_t sh = e - 130;
    uint32_t mag;
    if (sh >= 0) {
        mag = sh > 7 ? 0x7fffffffu : m << sh;
    } else {
        mag = sh <= -24 ? 0 : m >> -sh;
    }
    return (b & 0x80000000u) ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
}

// exp(-d) in Q31 for d >= 0 in Q20.
inline uint32_t exp_neg_q31(uint32_t d) {
    const uint64_t t = (static_cast<uint64_t>(d) * 1549082005u) >> 30;    // d * log2(e), Q20
    const uint64_t whole = t >> 20;
    if (whole >= 31) {
        return 0;
    }
    const uint32_t frac = static_cast<uint32_t>(t) & 0xfffffu;
    const uint32_t a = static_cast<uint32_t>((static_cast<uint64_t>(frac & 0x1fffu) * 1488522236u) >> 20);  // Q31
    const uint32_t poly = 0x80000000u - a + static_cast<uint32_t>((static_cast<uint64_t>(a) * a) >> 32);
    const uint32_t v = static_cast<uint32_t>((static_cast<uint64_t>(POW2_NEG128[frac >> 13]) * poly) >> 31);
    return v >> whole;
}

// Unsigned Q62 (value < 2) -> f32 bits, round to nearest even.
inline uint32_t q62_to_f32(uint64_t q) {
    if (q == 0) {
        return 0;
    }
    uint32_t top = 63;
    while (!((q >> top) & 1u)) {
        --top;
    }
    uint32_t exponent = top + 127 - 62;
    uint32_t mant;
    if (top >= 24) {
        const uint32_t drop = top - 23;
        mant = static_cast<uint32_t>(q >> drop);            // 24 bits incl. the leading one
        const uint64_t rest = q & ((uint64_t{1} << drop) - 1);
        const uint64_t half = uint64_t{1} << (drop - 1);
        if (rest > half || (rest == half && (mant & 1u))) {
            if (++mant == 0x1000000u) {
                mant = 0x800000u;
                ++exponent;
            }
        }
    } else {
        mant = static_cast<uint32_t>(q << (23 - top));
    }
    return (exponent << 23) | (mant & 0x7fffffu);
}

inline void route_row(const uint32_t* v, uint32_t* out) {
    uint32_t best_key[TOPK], best_id[TOPK];
    uint32_t count = 0, threshold = 0;
    for (uint32_t e = 0; e < N_EXP; ++e) {
        const uint32_t key = order_key(v[e]);
        if (count == TOPK && key <= threshold) {
            continue;
        }
        uint32_t p = count < TOPK ? count++ : TOPK - 1;
        while (p > 0 && best_key[p - 1] < key) {
            best_key[p] = best_key[p - 1];
            best_id[p] = best_id[p - 1];
            --p;
        }
        best_key[p] = key;
        best_id[p] = e;
        threshold = best_key[TOPK - 1];
    }

    const int32_t mx = f32_to_q20(v[best_id[0]]);
    uint32_t raw[TOPK];
    uint64_t total = 0;
    for (uint32_t k = 0; k < TOPK; ++k) {
        raw[k] = exp_neg_q31(static_cast<uint32_t>(mx - f32_to_q20(v[best_id[k]])));
        total += raw[k];
    }
    const uint64_t rec = (uint64_t{1} << 62) / total;       // total in [2^31, 10 * 2^31): Q31 reciprocal
    for (uint32_t k = 0; k < TOPK; ++k) {
        out[R_IDS + k] = best_id[k];
        out[R_W + k] = q62_to_f32(static_cast<uint64_t>(raw[k]) * rec);
    }
    // sigmoid(g) = 1 / (1 + e^-g): with q = exp(-|g|), g >= 0 -> 1 / (1 + q), else q / (1 + q)
    const int32_t g = f32_to_q20(v[N_EXP]);
    const uint32_t q = exp_neg_q31(static_cast<uint32_t>(g < 0 ? -g : g));
    const uint64_t inv = (uint64_t{1} << 62) / (uint64_t{0x80000000u} + q);
    out[R_SH] = q62_to_f32((g >= 0 ? uint64_t{0x80000000u} : uint64_t{q}) * inv);
    for (uint32_t k = R_SH + 1; k < ROUTE_WORDS; ++k) {
        out[k] = 0;
    }
}

}  // namespace fnx_moe
