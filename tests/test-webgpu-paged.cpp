// MUL_MAT_ID with paged expert weights (WebGPU_Paged) must equal the same op with the weights in an ordinary buffer.
// Two weight tensors routed by the same ids (as gate_up and down are) share one graph, so the backend makes both
// resident in one batch. Covers the one-token (vec) and gathered paths, slot counts from 1 to n_expert (passes when the routed experts
// outnumber the slots), a strided ids view (as top-k produces), and repeated runs that evict and reload experts.
// Then: repeated ids within a token, weight writes after experts are resident (set, async set, memset, clear),
// two backends sharing one paged weight buffer from two threads, and support checks that must not read other
// backends' buffers or reject a paged tensor whose full size exceeds a binding.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

typedef ggml_backend_buffer_type_t (*paged_buft_fn)(ggml_backend_dev_t, uint32_t, const ggml_webgpu_page_source *);

struct page_store {
    std::map<std::string, std::vector<uint8_t>> bytes;  // tensor name -> its data
    int                                          reads   = 0;
    int                                          writes  = 0;
    int                                          batches = 0;
};

static bool store_has(void * ud, const char * name, size_t nbytes) {
    auto * s  = (page_store *) ud;
    auto   it = s->bytes.find(name);
    return it != s->bytes.end() && it->second.size() == nbytes && s->writes > 0;
}

static void store_write(void * ud, const char * name, size_t off, const void * src, size_t n) {
    auto * s = (page_store *) ud;
    auto & b = s->bytes[name];
    if (b.size() < off + n) {
        b.resize(off + n);
    }
    memcpy(b.data() + off, src, n);
    s->writes++;
}

static void store_read(void * ud, const char * name, size_t off, void * dst, size_t n) {
    auto * s = (page_store *) ud;
    memcpy(dst, s->bytes.at(name).data() + off, n);
    s->reads++;
}

static void store_read_batch(void * ud, size_t n, const char * const * names, const size_t * offs, void * const * dsts,
                             const size_t * sizes) {
    ((page_store *) ud)->batches++;
    for (size_t i = 0; i < n; i++) {
        store_read(ud, names[i], offs[i], dsts[i], sizes[i]);
    }
}

struct config {
    ggml_type type;
    int       n_expert, n_used, n_tokens, k, m, n_slots;
    bool      source;  // a page source instead of the host-memory store
};

// A write to the weights between round 0 and round 1, after round 0 made experts resident.
enum weight_write { WRITE_NONE, WRITE_SET, WRITE_SET_ASYNC, WRITE_MEMSET, WRITE_CLEAR };
static const char * weight_write_name[] = { "none", "set", "set_async", "memset", "clear" };

// The second weight tensor: the same bytes, rows reversed, so its output differs from the first.
static std::vector<uint8_t> reversed_rows(const config & c, const std::vector<uint8_t> & as_data) {
    std::vector<uint8_t> out(as_data.size());
    const size_t         n_rows = (size_t) c.m * c.n_expert;
    const size_t         row    = as_data.size() / n_rows;
    for (size_t r = 0; r < n_rows; r++) {
        memcpy(out.data() + r * row, as_data.data() + (n_rows - 1 - r) * row, row);
    }
    return out;
}

// The graph of run(): two weight tensors routed by one strided ids view.
struct moe_graph {
    ggml_context * ci;
    ggml_tensor *  b, *ids_full, *out, *out2;
    ggml_cgraph *  gf;
    ggml_gallocr_t galloc;
};

static moe_graph build_graph(ggml_backend_t backend, const config & c, ggml_tensor * as, ggml_tensor * as2,
                             const std::vector<float> & b_data) {
    moe_graph        g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ci                = ggml_init(ip);
    g.b                 = ggml_new_tensor_3d(g.ci, GGML_TYPE_F32, c.k, c.n_used, c.n_tokens);
    // ids as a strided view, as top-k leaves it: [n_used, n_tokens] out of [n_expert, n_tokens]
    g.ids_full          = ggml_new_tensor_2d(g.ci, GGML_TYPE_I32, c.n_expert, c.n_tokens);
    ggml_tensor * ids   = ggml_view_2d(g.ci, g.ids_full, c.n_used, c.n_tokens, g.ids_full->nb[1], 0);
    g.out               = ggml_mul_mat_id(g.ci, as, g.b, ids);
    g.out2              = ggml_mul_mat_id(g.ci, as2, g.b, ids);
    g.gf                = ggml_new_graph(g.ci);
    ggml_build_forward_expand(g.gf, g.out);
    ggml_build_forward_expand(g.gf, g.out2);
    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(g.galloc, g.gf));
    ggml_backend_tensor_set(g.b, b_data.data(), 0, b_data.size() * sizeof(float));
    return g;
}

static std::vector<float> compute_round(ggml_backend_t backend, const config & c, moe_graph & g,
                                        const std::vector<int32_t> & r) {
    std::vector<int32_t> full(c.n_expert * c.n_tokens, -1);
    for (int t = 0; t < c.n_tokens; t++) {
        memcpy(full.data() + t * c.n_expert, r.data() + t * c.n_used, c.n_used * sizeof(int32_t));
    }
    ggml_backend_tensor_set(g.ids_full, full.data(), 0, full.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> o(ggml_nelements(g.out) + ggml_nelements(g.out2));
    ggml_backend_tensor_get(g.out, o.data(), 0, ggml_nbytes(g.out));
    ggml_backend_tensor_get(g.out2, o.data() + ggml_nelements(g.out), 0, ggml_nbytes(g.out2));
    return o;
}

static void free_graph(moe_graph & g) {
    ggml_gallocr_free(g.galloc);
    ggml_free(g.ci);
}

// Runs mul_mat_id(as, b, ids) on `backend` with `as` in `as_buft`; returns dst for each of `rounds` id sets.
// `write` changes the weights after round 0 (to `as_new`, or to zero bytes for memset and clear).
static std::vector<std::vector<float>> run(ggml_backend_t backend, ggml_backend_buffer_type_t as_buft, const config & c,
                                           const std::vector<uint8_t> & as_data, const std::vector<float> & b_data,
                                           const std::vector<std::vector<int32_t>> & rounds,
                                           weight_write write = WRITE_NONE, const std::vector<uint8_t> * as_new = nullptr) {
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   cw = ggml_init(wp);
    ggml_tensor *    as  = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
    ggml_tensor *    as2 = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
    ggml_set_name(as, "blk.0.ffn_up_exps.weight");
    ggml_set_name(as2, "blk.0.ffn_down_exps.weight");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(cw, as_buft);
    ggml_backend_tensor_set(as, as_data.data(), 0, as_data.size());
    std::vector<uint8_t> as2_data = reversed_rows(c, as_data);
    ggml_backend_tensor_set(as2, as2_data.data(), 0, as2_data.size());

    moe_graph g = build_graph(backend, c, as, as2, b_data);

    std::vector<std::vector<float>> results;
    for (size_t i = 0; i < rounds.size(); i++) {
        if (i == 1) {
            switch (write) {
                case WRITE_NONE:
                    break;
                case WRITE_SET:
                    ggml_backend_tensor_set(as, as_new->data(), 0, as_new->size());
                    break;
                case WRITE_SET_ASYNC:
                    ggml_backend_tensor_set_async(backend, as, as_new->data(), 0, as_new->size());
                    ggml_backend_synchronize(backend);
                    break;
                case WRITE_MEMSET:
                    ggml_backend_tensor_memset(as, 0, 0, ggml_nbytes(as));
                    break;
                case WRITE_CLEAR:
                    ggml_backend_buffer_clear(wbuf, 0);
                    break;
            }
        }
        results.push_back(compute_round(backend, c, g, rounds[i]));
    }
    free_graph(g);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cw);
    return results;
}

static size_t count_mismatches(const std::vector<std::vector<float>> & ref, const std::vector<std::vector<float>> & got,
                               double & max_diff) {
    size_t mismatches = 0;
    max_diff          = 0;
    for (size_t r = 0; r < ref.size(); r++) {
        for (size_t i = 0; i < ref[r].size(); i++) {
            if (memcmp(&ref[r][i], &got[r][i], sizeof(float)) != 0) {
                mismatches++;
                max_diff = std::max(max_diff, (double) std::fabs(ref[r][i] - got[r][i]));
            }
        }
    }
    return mismatches;
}

// Random weights quantized to the config's type.
static std::vector<uint8_t> make_weights(const config & c, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float>                    w((size_t) c.k * c.m * c.n_expert);
    for (float & x : w) {
        x = u(rng);
    }
    std::vector<uint8_t> as_data(ggml_row_size(c.type, c.k) * c.m * c.n_expert);
    if (c.type == GGML_TYPE_F32) {
        memcpy(as_data.data(), w.data(), as_data.size());
    } else {
        ggml_quantize_chunk(c.type, w.data(), as_data.data(), 0, (int64_t) c.m * c.n_expert, c.k, nullptr);
    }
    return as_data;
}

static std::vector<float> make_input(const config & c, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float>                    b((size_t) c.k * c.n_used * c.n_tokens);
    for (float & x : b) {
        x = u(rng);
    }
    return b;
}

// `n` rounds of routing: distinct experts per token as top-k gives, or with repeats when `repeats` is set.
static std::vector<std::vector<int32_t>> make_rounds(const config & c, std::mt19937 & rng, int n, bool repeats) {
    std::vector<std::vector<int32_t>> rounds;
    for (int r = 0; r < n; r++) {
        std::vector<int32_t> ids;
        for (int t = 0; t < c.n_tokens; t++) {
            std::vector<int32_t> perm(c.n_expert);
            for (int e = 0; e < c.n_expert; e++) {
                perm[e] = e;
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            if (repeats) {
                perm[1] = perm[0];  // e.g. [a, a, b, a]
                perm[3] = perm[0];
            }
            ids.insert(ids.end(), perm.begin(), perm.begin() + c.n_used);
        }
        rounds.push_back(ids);
    }
    return rounds;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("WebGPU");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-webgpu-paged: no WebGPU backend, skipping\n");
        return 0;
    }
    auto paged_buft = (paged_buft_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_webgpu_paged_buffer_type");
    GGML_ASSERT(paged_buft != nullptr);
    ggml_backend_dev_t dev     = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t     backend = ggml_backend_dev_init(dev, nullptr);

    std::vector<config> configs;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int n_tokens : { 1, 3, 32 }) {
            for (int n_slots : { 1, 3, 8, 16 }) {
                configs.push_back({ type, 16, 4, n_tokens, 256, 64, n_slots, false });
            }
        }
    }
    // K-quants (most published GGUFs): decode and a prefill with passes
    for (ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        configs.push_back({ type, 16, 4, 1, 256, 64, 8, false });
        configs.push_back({ type, 16, 4, 32, 256, 64, 3, false });
    }
    configs.push_back({ GGML_TYPE_Q4_0, 128, 8, 1, 512, 96, 8, true });   // Qwen3-like routing, decode
    configs.push_back({ GGML_TYPE_Q4_0, 128, 8, 17, 512, 96, 20, true });  // prefill: passes
    configs.push_back({ GGML_TYPE_Q8_0, 128, 8, 64, 256, 64, 128, true }); // every expert resident

    std::mt19937 rng(42);
    int          failed = 0;
    for (const config & c : configs) {
        std::vector<uint8_t>              as_data = make_weights(c, rng);
        std::vector<float>                b_data  = make_input(c, rng);
        std::vector<std::vector<int32_t>> rounds  = make_rounds(c, rng, 3, false);

        page_store              store;
        ggml_webgpu_page_source src = { &store, store_has, store_write, store_read, store_read_batch, nullptr };
        ggml_backend_buffer_type_t pbuft = paged_buft(dev, c.n_slots, c.source ? &src : nullptr);

        auto ref = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds);
        auto got = run(backend, pbuft, c, as_data, b_data, rounds);

        double     max_diff   = 0;
        size_t     mismatches = count_mismatches(ref, got, max_diff);
        const bool ok         = mismatches == 0 && (!c.source || (store.reads > 0 && store.batches > 0));
        failed += ok ? 0 : 1;
        printf("  %s %-5s experts=%3d used=%d tokens=%2d slots=%3d %s: %s (mismatches %zu, max diff %g, reads %d in %d batches)\n",
               ok ? "OK  " : "FAIL", ggml_type_name(c.type), c.n_expert, c.n_used, c.n_tokens, c.n_slots,
               c.source ? "source" : "ram   ", ok ? "identical" : "differs", mismatches, max_diff, store.reads, store.batches);
    }
    size_t n_checks = configs.size();

    // repeated ids within a token, with passes (3 slots) and without. Needs the duplicate-id MUL_MAT_ID shaders of
    // upstream #30451, which this branch predates (the plain path differs too), so it runs only when asked.
    for (int n_tokens : { 1, 5 }) {
        if (getenv("TEST_REPEATED_IDS") == nullptr) {
            break;
        }
        for (int n_slots : { 3, 16 }) {
            config c = { GGML_TYPE_Q4_0, 16, 4, n_tokens, 256, 64, n_slots, false };
            auto   as_data = make_weights(c, rng);
            auto   b_data  = make_input(c, rng);
            auto   rounds  = make_rounds(c, rng, 3, true);
            auto   ref     = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds);
            auto   got     = run(backend, paged_buft(dev, c.n_slots, nullptr), c, as_data, b_data, rounds);
            double max_diff;
            size_t mm = count_mismatches(ref, got, max_diff);
            failed += mm == 0 ? 0 : 1;
            n_checks++;
            printf("  %s repeated ids tokens=%d slots=%2d: mismatches %zu\n", mm == 0 ? "OK  " : "FAIL", n_tokens, n_slots, mm);
        }
    }

    // weight writes after round 0 made experts resident: round 1 must use the new bytes
    for (weight_write write : { WRITE_SET, WRITE_SET_ASYNC, WRITE_MEMSET, WRITE_CLEAR }) {
        for (bool source : { false, true }) {
            config c = { GGML_TYPE_Q8_0, 16, 4, 3, 256, 64, 8, source };
            auto   as_data = make_weights(c, rng);
            auto   as_new  = make_weights(c, rng);
            auto   b_data  = make_input(c, rng);
            auto   rounds  = make_rounds(c, rng, 1, false);
            rounds.push_back(rounds[0]);  // the same experts again, so a stale slot would be a hit

            page_store              store;
            ggml_webgpu_page_source src = { &store, nullptr, store_write, store_read, store_read_batch, nullptr };
            auto ref = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds, write, &as_new);
            auto got = run(backend, paged_buft(dev, c.n_slots, source ? &src : nullptr), c, as_data, b_data, rounds, write,
                           &as_new);
            double     max_diff;
            size_t     mm      = count_mismatches(ref, got, max_diff);
            const bool changed = memcmp(ref[0].data(), ref[1].data(), ref[0].size() * sizeof(float)) != 0;
            const bool ok      = mm == 0 && changed;
            failed += ok ? 0 : 1;
            n_checks++;
            printf("  %s weight write %-9s %s: mismatches %zu%s\n", ok ? "OK  " : "FAIL", weight_write_name[write],
                   source ? "source" : "ram   ", mm, changed ? "" : " (the write did not change the output)");
        }
    }

    // two backends share one paged weight buffer and compute from two threads
    {
        config c = { GGML_TYPE_Q4_0, 32, 4, 2, 256, 64, 4, false };
        auto   as_data = make_weights(c, rng);
        auto   b_data  = make_input(c, rng);
        std::vector<std::vector<int32_t>> rounds_a = make_rounds(c, rng, 40, false);
        std::vector<std::vector<int32_t>> rounds_b = make_rounds(c, rng, 40, false);
        auto ref_a = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds_a);
        auto ref_b = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds_b);

        ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   cw = ggml_init(wp);
        ggml_tensor *    as  = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
        ggml_tensor *    as2 = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
        ggml_set_name(as, "blk.0.ffn_up_exps.weight");
        ggml_set_name(as2, "blk.0.ffn_down_exps.weight");
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(cw, paged_buft(dev, c.n_slots, nullptr));
        ggml_backend_tensor_set(as, as_data.data(), 0, as_data.size());
        auto as2_data = reversed_rows(c, as_data);
        ggml_backend_tensor_set(as2, as2_data.data(), 0, as2_data.size());

        ggml_backend_t backend_b = ggml_backend_dev_init(dev, nullptr);
        moe_graph      ga        = build_graph(backend, c, as, as2, b_data);
        moe_graph      gb        = build_graph(backend_b, c, as, as2, b_data);
        std::vector<std::vector<float>> got_a, got_b;
        std::thread ta([&] {
            for (const auto & r : rounds_a) {
                got_a.push_back(compute_round(backend, c, ga, r));
            }
        });
        std::thread tb([&] {
            for (const auto & r : rounds_b) {
                got_b.push_back(compute_round(backend_b, c, gb, r));
            }
        });
        ta.join();
        tb.join();
        double max_diff;
        size_t mm = count_mismatches(ref_a, got_a, max_diff) + count_mismatches(ref_b, got_b, max_diff);
        failed += mm == 0 ? 0 : 1;
        n_checks++;
        printf("  %s two backends, shared paged weights, two threads x %zu rounds: mismatches %zu\n", mm == 0 ? "OK  " : "FAIL",
               rounds_a.size(), mm);
        free_graph(ga);
        free_graph(gb);
        ggml_backend_free(backend_b);
        ggml_backend_buffer_free(wbuf);
        ggml_free(cw);
    }

    // support checks: another backend's buffer must not be read as ours, and a paged tensor binds only its pool
    {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_init_params p  = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *   cc = ggml_init(p);
        ggml_tensor *    w  = ggml_new_tensor_2d(cc, GGML_TYPE_F32, 256, 64);
        ggml_tensor *    x  = ggml_new_tensor_2d(cc, GGML_TYPE_F32, 256, 4);
        ggml_tensor *    y  = ggml_mul_mat(cc, w, x);
        ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cc, cpu);
        ggml_backend_buffer_clear(cbuf, 0x00);
        const bool zeros = ggml_backend_dev_supports_op(dev, y);
        ggml_backend_buffer_clear(cbuf, 0xFF);
        const bool ones = ggml_backend_dev_supports_op(dev, y);
        const bool ok   = zeros == ones;
        failed += ok ? 0 : 1;
        n_checks++;
        printf("  %s support check ignores the bytes of a CPU buffer (zeros %d, ones %d)\n", ok ? "OK  " : "FAIL", zeros, ones);
        ggml_backend_buffer_free(cbuf);
        ggml_free(cc);
        ggml_backend_free(cpu);

        // the loader's probe: a zero-size buffer of the paged type, experts of 32 MiB each (8 GiB in all), 4 slots
        ggml_init_params      pp    = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context *        cp    = ggml_init(pp);
        ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(paged_buft(dev, 4, nullptr), 0);
        ggml_tensor *         big   = ggml_new_tensor_3d(cp, GGML_TYPE_F16, 4096, 4096, 256);
        big->buffer                 = probe;
        ggml_tensor * bx            = ggml_new_tensor_3d(cp, GGML_TYPE_F32, 4096, 1, 1);
        ggml_tensor * bids          = ggml_new_tensor_2d(cp, GGML_TYPE_I32, 1, 1);
        ggml_tensor * by            = ggml_mul_mat_id(cp, big, bx, bids);
        const bool    leaf          = ggml_backend_dev_supports_op(dev, big);
        const bool    op            = ggml_backend_dev_supports_op(dev, by);
        failed += leaf && op ? 0 : 1;
        n_checks++;
        printf("  %s paged 8 GiB tensor with a 128 MiB pool: leaf %d, mul_mat_id %d\n", leaf && op ? "OK  " : "FAIL", leaf, op);
        ggml_backend_buffer_free(probe);
        ggml_free(cp);
    }

    printf("%zu/%zu checks passed\n", n_checks - failed, n_checks);
    ggml_backend_free(backend);
    return failed == 0 ? 0 : 1;
}
