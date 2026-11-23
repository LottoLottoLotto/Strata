// fnx_cpu.cpp - Qwen3.8-Flash-Next routed experts on the host CPU for the Tenstorrent port, through Strata's own
// CPU kernels (strata_kernels_cpu + ggml-cpu), with the arithmetic Strata's CPU pool uses for a native pack:
//
//   gate/up  native_quant_act(x) -> native_gu_rows  (ff = silu(gate . a) * (up . a))       pool.cpp mode 5
//   down     Q2_0 down:  act_quant_any(ff) -> q2_rows_any                                  pool.cpp:649-654, 769
//            otherwise:  native_quant_h(ff) -> native_down_rows                             pool.cpp:655-658, 770
//
// Weights are read in place from the mapped GGUF (the caller passes each layer's gate/up/down tensor base): a GGUF
// expert stack is ne = [n_in, n_out, n_expert], so expert e's rows start at base + e * n_out * row_bytes. The
// kernels address a blob as [gate | up | down] through NativeFmt::up_off / down_off, so each expert gets a NativeFmt
// whose offsets are the distances from its gate rows to its up and down rows (unsigned, modulo 2^64: the tensors
// may lie in either order in the mapping; the flat x86-64 address space makes the wrapped sum the right address).
//
// Output rows are UNWEIGHTED (as the engine's CPU pool returns them); the router weights and the shared expert
// are applied by the caller.  Tokens of one expert run together in groups of at most MAXT (8), or of at most
// kIq512WideMax (32) where the layer's gate/up runs on the AVX-512 i-quant kernel (native_gu_max_tokens: its wide
// path decodes each weight block once for all of them; a prompt chunk's expert often has more than 8 tokens); the
// down rows of a wide group run in balanced pieces of at most MAXT.  With STRATA_IQ_MT_MIN=1 a token's rows do not
// depend on that grouping (#152).
//
// Threads (fnx_init): n_threads - 1 pool workers plus the thread that runs a call (the caller of fnx_moe, or the
// leader thread of fnx_moe_submit), pinned one per physical core as Strata's ExpertPool pins them (pool.cpp
// detect_cpu_topology: two workers on SMT siblings share the core's load/store bandwidth), CPU 0's core last
// (cpu_order).  An idle thread spins on _mm_pause for 20 ms (Strata's kSpinBeforeSleep) before it sleeps, yielding
// its CPU now and then to the process's other threads (Doorbell), so within a decode step (a call every ~1 ms) no
// call waits for a wake-up.  A parallel region completes when its TASKS are done (not when every worker has seen
// it).  fnx_moe_submit hands a call to the spinning leader thread and returns at once (the caller dispatches device
// work meanwhile); fnx_moe_wait spins until it is done.  The work split does not depend on which threads run it, and
// the scratch buffers persist across calls (no allocation once warm), so every output bit is as before.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace k = strata::kernels::cpu;

namespace {

using Clock = std::chrono::steady_clock;

struct Layer {
    bool set = false;
    k::NativeFmt f;
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    size_t gu_stride = 0, d_stride = 0;   // bytes per expert in the gate/up and down stacks
};

std::vector<Layer> g_layers;
int g_threads = 1;

void set_err(char* err, int errlen, const std::string& s) {
    if (err && errlen > 0) std::snprintf(err, (size_t) errlen, "%s", s.c_str());
}

// ---- CPU placement ------------------------------------------------------------------------------------------------
long topo_read(int cpu, const char* what) {
    char path[96];
    std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    std::FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

// The CPUs the leader and the workers are pinned to, in that order.  FNX_CPU_LIST ("1,2,3...") when set; else the
// CPUs this process may run on: the first allowed CPU of each physical core (package, core_id), then the SMT
// siblings (Strata's pool.cpp order: two workers on one core share its load/store bandwidth), and the core of CPU 0
// last of all (as Strata's physical_cores(skip_first) keeps it for its host thread): with 16 threads on 16 cores the
// process's main thread, which dispatches the device and waits for its read-backs, keeps a core of its own.  Without
// sysfs every allowed CPU counts as a core.
std::vector<int> cpu_order() {
    std::vector<int> out;
    if (const char* e = std::getenv("FNX_CPU_LIST")) {
        for (const char* p = e; *p;) {
            char* end = nullptr;
            const long c = std::strtol(p, &end, 10);
            if (end == p) break;
            out.push_back((int) c);
            p = *end == ',' ? end + 1 : end;
        }
        return out;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    const std::pair<long, long> core0{topo_read(0, "physical_package_id"), topo_read(0, "core_id")};
    std::vector<int> siblings, last;
    std::vector<std::pair<long, long>> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::pair<long, long> key{topo_read(c, "physical_package_id"), topo_read(c, "core_id")};
        const bool known = key.first >= 0 && key.second >= 0;
        if (c == 0 || (known && key == core0)) {
            last.push_back(c);
        } else if (!known || std::find(seen.begin(), seen.end(), key) == seen.end()) {
            if (known) seen.push_back(key);
            out.push_back(c);
        } else {
            siblings.push_back(c);
        }
    }
    out.insert(out.end(), siblings.begin(), siblings.end());
    out.insert(out.end(), last.begin(), last.end());
    return out;
}

void pin_to(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

// How long an idle thread spins before it sleeps (FNX_CPU_SPIN_US overrides the 20 ms).
std::chrono::microseconds spin_time() {
    if (const char* e = std::getenv("FNX_CPU_SPIN_US")) return std::chrono::microseconds(std::max(0, std::atoi(e)));
    return std::chrono::milliseconds(20);
}

// A counter threads wait on: spin (_mm_pause) while it equals what they saw, sleep after `spin`.  Every 1024 pauses
// the spinner calls sched_yield, so a runnable thread of the process that shares its CPU (numpy's OpenBLAS threads:
// the QSA host indexer's gemv runs 32 of them, which then spin-yield ~50 ms) gets it: measured on traced decode, a
// pure _mm_pause spin made that indexer 1.3 -> 15 ms per step and plain decode slower than before (12.5 tok/s); with
// the yield 1.5 ms (16.4 tok/s).  Lost wake-ups are impossible for the reason Strata's ExpertPool::publish gives: the
// ringer bumps `value` and then reads `sleepers`, a sleeper bumps `sleepers` and then reads `value`, both seq_cst, so
// one of them sees the other's write.
struct Doorbell {
    alignas(64) std::atomic<uint32_t> value{0};
    alignas(64) std::atomic<uint32_t> sleepers{0};
    std::mutex mu;
    std::condition_variable cv;

    void ring() {
        value.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
    }
    // The first value != seen (acquire), or seen once `stop` is set.
    uint32_t wait(uint32_t seen, const std::atomic<bool>& stop, std::chrono::microseconds spin) {
        const auto t0 = Clock::now();
        for (uint32_t n = 0;;) {
            const uint32_t v = value.load(std::memory_order_acquire);
            if (v != seen) return v;
            if (stop.load(std::memory_order_relaxed)) return seen;
            _mm_pause();
            if ((++n & 1023u) != 0) continue;
            sched_yield();
            if (Clock::now() - t0 < spin) continue;
            std::unique_lock<std::mutex> lk(mu);
            sleepers.fetch_add(1, std::memory_order_seq_cst);
            cv.wait(lk, [&] {
                return value.load(std::memory_order_seq_cst) != seen || stop.load(std::memory_order_relaxed);
            });
            sleepers.fetch_sub(1, std::memory_order_relaxed);
        }
    }
};

// Persistent workers.  A region publishes (epoch, n tasks, next task) in one word that every thread claims from by
// CAS, so a worker that wakes late cannot claim a task of a later region (Strata's ExpertPool::claim, issue #29), and
// the region is over when n tasks are DONE: the caller never waits for a worker that found nothing left to do.
class Pool {
public:
    using Fn = void (*)(void*, int);

    Pool(int n_workers, const std::vector<int>& cpus, std::chrono::microseconds spin) : spin_(spin) {
        for (int i = 0; i < n_workers; ++i) {
            const int cpu = i < (int) cpus.size() ? cpus[(size_t) i] : -1;
            th_.emplace_back([this, cpu] {
                pin_to(cpu);
                loop();
            });
        }
    }
    ~Pool() {
        stop_.store(true, std::memory_order_release);
        bell_.ring();
        for (auto& t : th_) t.join();
    }

    // fn(ctx, task) for task in [0, n), on the workers and the calling thread; returns when every task is done.
    void run(int n, Fn fn, void* ctx) {
        for (int base = 0; base < n; base += kMaxTasks) {
            const uint32_t m = (uint32_t) std::min(n - base, kMaxTasks);
            fn_ = fn;
            ctx_ = ctx;
            base_ = base;
            done_.store(0, std::memory_order_relaxed);   // every task of the last region is done, none can be claimed
            const uint32_t e = bell_.value.load(std::memory_order_relaxed) + 1;   // only the caller rings
            head_.store((uint64_t) e << 32 | (uint64_t) m << 16, std::memory_order_release);
            bell_.ring();
            drain(e);
            while (done_.load(std::memory_order_acquire) < m) _mm_pause();
        }
    }

private:
    static constexpr int kMaxTasks = 0xffff;

    int claim(uint32_t e) {
        uint64_t h = head_.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (h >> 32) != e) return -1;                   // not the region this thread woke for
            const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
            if (i >= n) return -1;                                       // every task claimed
            if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
                return (int) i;
        }
    }
    // fn_ / ctx_ / base_ are read only after a successful claim: the caller wrote them before the head_ release and
    // rewrites them only after every task of this region is done.
    void drain(uint32_t e) {
        for (int i; (i = claim(e)) >= 0;) {
            fn_(ctx_, base_ + i);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    void loop() {
        uint32_t seen = 0;
        for (;;) {
            seen = bell_.wait(seen, stop_, spin_);
            if (stop_.load(std::memory_order_acquire)) return;
            drain(seen);
        }
    }

    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    int base_ = 0;
    Doorbell bell_;
    std::atomic<bool> stop_{false};
    std::chrono::microseconds spin_;
    std::vector<std::thread> th_;
};

std::unique_ptr<Pool> g_pool;

// Run fn(task) for task in [0, n) on the pool (the calling thread included).
template <class F>
void parallel(int n, F&& fn) {
    if (!g_pool || n <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    using L = std::remove_reference_t<F>;
    g_pool->run(n, [](void* c, int i) { (*static_cast<L*>(c))(i); },
                const_cast<void*>(static_cast<const void*>(&fn)));
}

// One group: up to kIq512WideMax tokens of one expert (MAXT unless the gate/up kernel takes more).
constexpr int WMAX = k::kIq512WideMax;
struct Group {
    int expert = 0, nt = 0;
    int tok[WMAX];         // token index
    int slot[WMAX];        // which of the token's k routed slots
    k::NativeFmt f;        // the layer's format with this expert's offsets
    const uint8_t* blob = nullptr;   // the expert's gate rows
    const uint8_t* down = nullptr;   // the expert's down rows
};

// Scratch of a call, kept across calls (one call runs at a time).  Grown, never shrunk; nothing reads an element
// a call has not written (ActQ::bp_pairs is reset before each quantization, as a fresh ActQ has it).
struct Scratch {
    std::vector<uint8_t> act, hq;
    std::vector<k::ActQ> actq, aq;
    std::vector<float> ff, parts;
    std::vector<Group> groups;
    std::vector<int> open = std::vector<int>(4096, -1);   // expert -> its open group (ids are < n_expert <= 4096)
};
Scratch g_scratch;

template <class T>
T* grow(std::vector<T>& v, size_t n) {
    if (v.size() < n) v.resize(n);
    return v.data();
}

// The weighted sum of one token's kk slot outputs, sum = (((0 + p0 w0) + p1 w1) + ...) + p(kk-1) w(kk-1): every
// product and sum rounded on its own (this file is built without FMA contraction), the order and the bits of
// numpy's einsum("tkd,tk->td") that fnx.tt.moe used on the host.
void weighted_sum(const float* parts, const float* wt, int kk, float* sum) {
    for (int d = 0; d < k::H; ++d) sum[d] = 0.f;
    for (int s = 0; s < kk; ++s) {
        const float w = wt[s];
        const float* p = parts + (size_t) s * k::H;
        for (int d = 0; d < k::H; ++d) sum[d] = sum[d] + p[d] * w;
    }
}

// out: each slot's unweighted output; or, with wt (nt, kk) and sum (nt, 2560), out == nullptr and only each token's
// weighted sum is written (the slot outputs go to scratch).
int moe_call(int layer, int nt, int kk, const float* x, const int32_t* ids, float* out, const float* wt, float* sum,
             char* err, int errlen) {
    if (sum != nullptr) out = grow(g_scratch.parts, (size_t) nt * kk * k::H);
    if (layer < 0 || layer >= (int) g_layers.size() || !g_layers[(size_t) layer].set) {
        set_err(err, errlen, "layer not registered");
        return -1;
    }
    const Layer& L = g_layers[(size_t) layer];
    const k::NativeFmt& f = L.f;
    const bool q2_down = k::q2_native_kernels(f.d_type);
    // Q2_0 gate/up (the MTP layer's experts): Strata's own Q2_0 rows on its ActQ activations, then SwiGLU, exactly
    // as its CPU pool computes a native Q2_0 layer (pool.cpp:633-644)
    const bool q2_gu = k::q2_native_kernels(f.gu_type);
    Scratch& S = g_scratch;

    // 1. the gate/up activation of every token (one per token: every expert of the layer shares the format)
    uint8_t* act = nullptr;
    k::ActQ* actq = nullptr;
    if (q2_gu) {
        actq = grow(S.actq, (size_t) nt);
        parallel(nt, [&](int t) {
            actq[t].bp_pairs = 0;
            k::act_quant_any(x + (size_t) t * k::H, k::H, actq[t]);
        });
    } else {
        act = grow(S.act, (size_t) nt * f.act_bytes);
        parallel(nt, [&](int t) { k::native_quant_act(f, x + (size_t) t * k::H, act + (size_t) t * f.act_bytes); });
    }

    // 2. group the slots by expert (first-seen order), at most `cap` tokens a group; skipped slots are zeroed
    std::vector<Group>& groups = S.groups;
    std::vector<int>& open = S.open;
    const int cap = q2_gu ? k::MAXT : k::native_gu_max_tokens(f.gu_type);
    groups.clear();
    for (int t = 0; t < nt; ++t)
        for (int s = 0; s < kk; ++s) {
            const int e = ids[(size_t) t * kk + s];
            if (e < 0) {
                std::memset(out + ((size_t) t * kk + s) * k::H, 0, sizeof(float) * k::H);
                continue;
            }
            if (e >= (int) open.size()) {
                for (const Group& g : groups) open[(size_t) g.expert] = -1;
                set_err(err, errlen, "expert id out of range");
                return -1;
            }
            int gi = open[(size_t) e];
            if (gi < 0 || groups[(size_t) gi].nt == cap) {
                Group g;
                g.expert = e;
                g.f = f;
                g.blob = L.gate + (size_t) e * L.gu_stride;
                const uint8_t* up = L.up + (size_t) e * L.gu_stride;
                g.down = L.down + (size_t) e * L.d_stride;
                g.f.up_off = (size_t) ((uintptr_t) up - (uintptr_t) g.blob);
                g.f.down_off = (size_t) ((uintptr_t) g.down - (uintptr_t) g.blob);
                groups.push_back(g);
                gi = (int) groups.size() - 1;
                open[(size_t) e] = gi;
            }
            Group& g = groups[(size_t) gi];
            g.tok[g.nt] = t;
            g.slot[g.nt] = s;
            ++g.nt;
        }
    for (const Group& g : groups) open[(size_t) g.expert] = -1;
    const int ng = (int) groups.size();
    if (ng == 0) return 0;

    // 3. gate/up rows, split by row ranges across all groups (about three tasks per thread, as the pool does)
    float* ff = grow(S.ff, (size_t) ng * WMAX * k::FF);
    auto ffp = [&](int gi, int j) { return ff + ((size_t) gi * WMAX + j) * k::FF; };
    const int parts_gu = std::max(1, (3 * g_threads + ng - 1) / ng);
    parallel(ng * parts_gu, [&](int task) {
        const int gi = task / parts_gu, p = task % parts_gu;
        const Group& g = groups[(size_t) gi];
        const int r0 = k::FF * p / parts_gu, r1 = k::FF * (p + 1) / parts_gu;
        float* o[WMAX];
        for (int j = 0; j < g.nt; ++j) o[j] = ffp(gi, j);
        if (q2_gu) {
            float gbuf[k::MAXT][k::FF], ubuf[k::MAXT][k::FF];
            float* gp[k::MAXT];
            float* up[k::MAXT];
            const k::ActQ* a[k::MAXT];
            for (int j = 0; j < g.nt; ++j) {
                gp[j] = gbuf[j];
                up[j] = ubuf[j];
                a[j] = &actq[(size_t) g.tok[j]];
            }
            const int nbk = (int) (f.n_embd / 64);
            k::q2_rows_any(g.blob, f.gu_row, nbk, a, g.nt, gp, r0, r1);
            k::q2_rows_any(g.blob + g.f.up_off, f.gu_row, nbk, a, g.nt, up, r0, r1);
            for (int j = 0; j < g.nt; ++j)
                for (int r = r0; r < r1; ++r) o[j][r] = (gbuf[j][r] / (1.f + std::exp(-gbuf[j][r]))) * ubuf[j][r];
            return;
        }
        const void* a[WMAX];
        for (int j = 0; j < g.nt; ++j) a[j] = act + (size_t) g.tok[j] * f.act_bytes;
        k::native_gu_rows(g.f, g.blob, a, g.nt, o, r0, r1);
    });

    // 4. the down activation of each (group, token)
    uint8_t* hq = nullptr;
    k::ActQ* aq = nullptr;
    if (q2_down) aq = grow(S.aq, (size_t) ng * WMAX);
    else hq = grow(S.hq, (size_t) ng * WMAX * f.h_bytes);
    parallel(ng, [&](int gi) {
        for (int j = 0; j < groups[(size_t) gi].nt; ++j) {
            if (q2_down) {
                k::ActQ& a = aq[(size_t) gi * WMAX + j];
                a.bp_pairs = 0;
                k::act_quant_any(ffp(gi, j), k::FF, a);
            } else {
                k::native_quant_h(f, ffp(gi, j), hq + ((size_t) gi * WMAX + j) * f.h_bytes);
            }
        }
    });

    // 5. down rows straight into the output slots (a wide group in balanced pieces of at most MAXT tokens: the down
    // kernels take MAXT)
    const int parts_d = std::max(1, (3 * g_threads + ng - 1) / ng);
    parallel(ng * parts_d, [&](int task) {
        const int gi = task / parts_d, p = task % parts_d;
        const Group& g = groups[(size_t) gi];
        const int r0 = k::H * p / parts_d, r1 = k::H * (p + 1) / parts_d;
        const int pieces = (g.nt + k::MAXT - 1) / k::MAXT;
        for (int c = 0; c < pieces; ++c) {
            const int j0 = g.nt * c / pieces, n = g.nt * (c + 1) / pieces - j0;
            float* o[k::MAXT];
            for (int j = 0; j < n; ++j) o[j] = out + ((size_t) g.tok[j0 + j] * kk + g.slot[j0 + j]) * k::H;
            if (q2_down) {
                const k::ActQ* a2[k::MAXT];
                for (int j = 0; j < n; ++j) a2[j] = &aq[(size_t) gi * WMAX + j0 + j];
                k::q2_rows_any(g.down, f.d_row, (int) (f.n_ff / 64), a2, n, o, r0, r1);
            } else {
                const void* h[k::MAXT];
                for (int j = 0; j < n; ++j) h[j] = hq + ((size_t) gi * WMAX + j0 + j) * f.h_bytes;
                k::native_down_rows(g.f, g.blob, h, n, o, r0, r1);
            }
        }
    });
    // 6. the weighted sums, a token per task
    if (sum != nullptr)
        parallel(nt, [&](int t) {
            weighted_sum(out + (size_t) t * kk * k::H, wt + (size_t) t * kk, kk, sum + (size_t) t * k::H);
        });
    return 0;
}

// The thread fnx_moe_submit hands calls to (it then runs the call's regions with the pool, as fnx_moe's caller does).
class Leader {
public:
    Leader(int cpu, std::chrono::microseconds spin) : spin_(spin) {
        th_ = std::thread([this, cpu] {
            pin_to(cpu);
            loop();
        });
    }
    ~Leader() {
        wait();
        stop_.store(true, std::memory_order_release);
        bell_.ring();
        th_.join();
    }
    void submit(int layer, int nt, int kk, const float* x, const int32_t* ids, float* out, const float* wt = nullptr,
                float* sum = nullptr) {
        wait();                                   // a call submitted and never waited for (its caller failed)
        job_ = Job{layer, nt, kk, x, ids, out, wt, sum};
        bell_.ring();
    }
    // Spin until the last submitted call is done; its (return code, microseconds), err filled on failure.
    std::pair<int, double> wait(char* err = nullptr, int errlen = 0) const {
        const uint32_t s = bell_.value.load(std::memory_order_relaxed);
        while (done_.load(std::memory_order_acquire) != s) _mm_pause();
        if (rc_ != 0) set_err(err, errlen, err_);
        return {rc_, us_};
    }

private:
    struct Job {
        int layer = 0, nt = 0, kk = 0;
        const float* x = nullptr;
        const int32_t* ids = nullptr;
        float* out = nullptr;
        const float* wt = nullptr;
        float* sum = nullptr;
    };
    void loop() {
        uint32_t seen = 0;
        for (;;) {
            seen = bell_.wait(seen, stop_, spin_);
            if (stop_.load(std::memory_order_acquire)) return;
            const auto t0 = Clock::now();
            const Job& j = job_;
            rc_ = moe_call(j.layer, j.nt, j.kk, j.x, j.ids, j.out, j.wt, j.sum, err_, (int) sizeof err_);
            us_ = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            done_.store(seen, std::memory_order_release);
        }
    }

    Doorbell bell_;                                // value = calls submitted
    alignas(64) std::atomic<uint32_t> done_{0};    // the last call done
    std::atomic<bool> stop_{false};
    Job job_;
    int rc_ = 0;
    double us_ = 0;
    char err_[512] = {};
    std::chrono::microseconds spin_;
    std::thread th_;
};

std::unique_ptr<Leader> g_leader;

}  // namespace

extern "C" {

/// Prepare for `n_layers` layers and use `n_threads` threads (the caller included).  Returns 0, or -1 with `err`.
/// The leader and the n_threads - 1 workers are pinned to cpu_order()'s first CPUs; FNX_CPU_PIN=0 leaves them
/// unpinned, FNX_CPU_SPIN_US sets their spin before sleeping (default 20000).
int fnx_init(int n_layers, int n_threads, char* err, int errlen) {
    if (!k::native_experts_available()) {
        set_err(err, errlen, "this build has no ggml-cpu native expert path");
        return -1;
    }
    if (!k::cpu_avx2_ok()) {
        set_err(err, errlen, "this CPU lacks AVX2 (the floor of Strata's expert kernels)");
        return -1;
    }
    g_leader.reset();   // finishes a pending call first
    g_pool.reset();
    g_layers.assign((size_t) std::max(0, n_layers), Layer{});
    g_threads = std::max(1, n_threads);
    std::vector<int> cpus = cpu_order();
    const char* pin = std::getenv("FNX_CPU_PIN");
    if (pin && std::strcmp(pin, "0") == 0) cpus.clear();
    const auto spin = spin_time();
    g_leader = std::make_unique<Leader>(cpus.empty() ? -1 : cpus[0], spin);
    if (g_threads > 1) {
        std::vector<int> wcpus(cpus.begin() + (cpus.empty() ? 0 : 1), cpus.end());
        g_pool = std::make_unique<Pool>(g_threads - 1, wcpus, spin);
    }
    return 0;
}

/// Register one layer's expert tensors: ggml types and the base addresses of the three stacks.
int fnx_set_layer(int layer, int gu_type, int d_type, int n_embd, int n_ff, const uint8_t* gate,
                  const uint8_t* up, const uint8_t* down, char* err, int errlen) {
    if (layer < 0 || layer >= (int) g_layers.size()) {
        set_err(err, errlen, "layer out of range");
        return -1;
    }
    if (n_embd != k::H || n_ff != k::FF) {
        set_err(err, errlen, "Strata's CPU kernels are built for n_embd 2560, n_ff 640");
        return -1;
    }
    if (g_leader) g_leader->wait();
    Layer& L = g_layers[(size_t) layer];
    std::string e;
    if (!k::native_fmt(gu_type, d_type, n_embd, n_ff, L.f, e)) {
        set_err(err, errlen, "layer " + std::to_string(layer) + ": " + e);
        return -1;
    }
    L.gate = gate;
    L.up = up;
    L.down = down;
    L.gu_stride = (size_t) n_ff * L.f.gu_row;
    L.d_stride = (size_t) n_embd * L.f.d_row;
    L.set = true;
    return 0;
}

/// Routed experts of one layer for `nt` tokens, on the calling thread and the pool.
///   x    (nt, 2560) f32      the MoE input of each token (after the layer's FFN norm)
///   ids  (nt, kk) int32      routed expert per slot; a negative id is skipped (computed elsewhere, e.g. on device)
///   out  (nt, kk, 2560) f32  each slot's UNWEIGHTED expert output; skipped slots are set to zero
int fnx_moe(int layer, int nt, int kk, const float* x, const int32_t* ids, float* out, char* err, int errlen) {
    if (g_leader) g_leader->wait();
    return moe_call(layer, nt, kk, x, ids, out, nullptr, nullptr, err, errlen);
}

/// fnx_moe on the leader thread: returns at once (-1 with `err` for an unregistered layer); x, ids and out must
/// stay valid until fnx_moe_wait returns.  A call not waited for is finished before the next one starts.
int fnx_moe_submit(int layer, int nt, int kk, const float* x, const int32_t* ids, float* out, char* err,
                   int errlen) {
    if (!g_leader) {
        set_err(err, errlen, "fnx_init first");
        return -1;
    }
    if (layer < 0 || layer >= (int) g_layers.size() || !g_layers[(size_t) layer].set) {
        set_err(err, errlen, "layer not registered");
        return -1;
    }
    g_leader->submit(layer, nt, kk, x, ids, out);
    return 0;
}

/// fnx_moe_submit returning each token's router-weighted sum instead of the slot outputs:
///   wt   (nt, kk) f32        the slots' weights (0 for a slot computed elsewhere)
///   sum  (nt, 2560) f32      sum over the slots of weight * expert output (numpy einsum's order and bits)
int fnx_moe_submit_sum(int layer, int nt, int kk, const float* x, const int32_t* ids, const float* wt, float* sum,
                       char* err, int errlen) {
    if (!g_leader) {
        set_err(err, errlen, "fnx_init first");
        return -1;
    }
    if (layer < 0 || layer >= (int) g_layers.size() || !g_layers[(size_t) layer].set) {
        set_err(err, errlen, "layer not registered");
        return -1;
    }
    g_leader->submit(layer, nt, kk, x, ids, nullptr, wt, sum);
    return 0;
}

/// Wait for the last fnx_moe_submit call (spinning): its return code (err filled on failure); *us = its time on the
/// leader thread in microseconds.
int fnx_moe_wait(double* us, char* err, int errlen) {
    if (!g_leader) {
        set_err(err, errlen, "fnx_init first");
        return -1;
    }
    const auto [rc, t] = g_leader->wait(err, errlen);
    if (us) *us = t;
    return rc;
}

}  // extern "C"
