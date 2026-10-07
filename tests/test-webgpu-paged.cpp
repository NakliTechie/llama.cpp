// MUL_MAT_ID with paged expert weights (WebGPU_Paged) must equal the same op with the weights in an ordinary buffer.
// Two weight tensors routed by the same ids (as gate_up and down are) share one graph, so the backend makes both
// resident in one batch. Covers the one-token (vec) and gathered paths, slot counts from 1 to n_expert (passes when the routed experts
// outnumber the slots), a strided ids view (as top-k produces), and repeated runs that evict and reload experts.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
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

// Runs mul_mat_id(as, b, ids) on `backend` with `as` in `as_buft`; returns dst for each of `rounds` id sets.
static std::vector<std::vector<float>> run(ggml_backend_t backend, ggml_backend_buffer_type_t as_buft, const config & c,
                                           const std::vector<uint8_t> & as_data, const std::vector<float> & b_data,
                                           const std::vector<std::vector<int32_t>> & rounds) {
    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   cw = ggml_init(wp);
    ggml_tensor *    as  = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
    ggml_tensor *    as2 = ggml_new_tensor_3d(cw, c.type, c.k, c.m, c.n_expert);
    ggml_set_name(as, "blk.0.ffn_up_exps.weight");
    ggml_set_name(as2, "blk.0.ffn_down_exps.weight");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(cw, as_buft);
    ggml_backend_tensor_set(as, as_data.data(), 0, as_data.size());
    // the second tensor: the same bytes, rows reversed, so its output differs from the first
    std::vector<uint8_t> as2_data(as_data.size());
    const size_t         row = as_data.size() / ((size_t) c.m * c.n_expert);
    for (size_t r = 0; r < (size_t) c.m * c.n_expert; r++) {
        memcpy(as2_data.data() + r * row, as_data.data() + ((size_t) c.m * c.n_expert - 1 - r) * row, row);
    }
    ggml_backend_tensor_set(as2, as2_data.data(), 0, as2_data.size());

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ci = ggml_init(ip);
    ggml_tensor *    b  = ggml_new_tensor_3d(ci, GGML_TYPE_F32, c.k, c.n_used, c.n_tokens);
    // ids as a strided view, as top-k leaves it: [n_used, n_tokens] out of [n_expert, n_tokens]
    ggml_tensor *    ids_full = ggml_new_tensor_2d(ci, GGML_TYPE_I32, c.n_expert, c.n_tokens);
    ggml_tensor *    ids      = ggml_view_2d(ci, ids_full, c.n_used, c.n_tokens, ids_full->nb[1], 0);
    ggml_tensor *    out      = ggml_mul_mat_id(ci, as, b, ids);
    ggml_tensor *    out2     = ggml_mul_mat_id(ci, as2, b, ids);
    ggml_cgraph *    gf       = ggml_new_graph(ci);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, out2);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    std::vector<std::vector<float>> results;
    for (const auto & r : rounds) {
        std::vector<int32_t> full(c.n_expert * c.n_tokens, -1);
        for (int t = 0; t < c.n_tokens; t++) {
            memcpy(full.data() + t * c.n_expert, r.data() + t * c.n_used, c.n_used * sizeof(int32_t));
        }
        ggml_backend_tensor_set(ids_full, full.data(), 0, full.size() * sizeof(int32_t));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        std::vector<float> o(ggml_nelements(out) + ggml_nelements(out2));
        ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
        ggml_backend_tensor_get(out2, o.data() + ggml_nelements(out), 0, ggml_nbytes(out2));
        results.push_back(std::move(o));
    }
    ggml_gallocr_free(galloc);
    ggml_free(ci);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cw);
    return results;
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
    configs.push_back({ GGML_TYPE_Q4_0, 128, 8, 1, 512, 96, 8, true });   // Qwen3-like routing, decode
    configs.push_back({ GGML_TYPE_Q4_0, 128, 8, 17, 512, 96, 20, true });  // prefill: passes
    configs.push_back({ GGML_TYPE_Q8_0, 128, 8, 64, 256, 64, 128, true }); // every expert resident

    std::mt19937 rng(42);
    int          failed = 0;
    for (const config & c : configs) {
        // weights: random f32, quantized to the test type
        const size_t       n_el = (size_t) c.k * c.m * c.n_expert;
        std::vector<float> w(n_el);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (float & x : w) {
            x = u(rng);
        }
        std::vector<uint8_t> as_data(ggml_row_size(c.type, c.k) * c.m * c.n_expert);
        if (c.type == GGML_TYPE_F32) {
            memcpy(as_data.data(), w.data(), as_data.size());
        } else {
            ggml_quantize_chunk(c.type, w.data(), as_data.data(), 0, (int64_t) c.m * c.n_expert, c.k, nullptr);
        }
        std::vector<float> b_data((size_t) c.k * c.n_used * c.n_tokens);
        for (float & x : b_data) {
            x = u(rng);
        }
        // 3 rounds of routing: distinct experts per token, as top-k gives
        std::vector<std::vector<int32_t>> rounds;
        for (int r = 0; r < 3; r++) {
            std::vector<int32_t> ids;
            for (int t = 0; t < c.n_tokens; t++) {
                std::vector<int32_t> perm(c.n_expert);
                for (int e = 0; e < c.n_expert; e++) {
                    perm[e] = e;
                }
                std::shuffle(perm.begin(), perm.end(), rng);
                ids.insert(ids.end(), perm.begin(), perm.begin() + c.n_used);
            }
            rounds.push_back(ids);
        }

        page_store              store;
        ggml_webgpu_page_source src = { &store, store_has, store_write, store_read, store_read_batch };
        ggml_backend_buffer_type_t pbuft = paged_buft(dev, c.n_slots, c.source ? &src : nullptr);

        auto ref = run(backend, ggml_backend_dev_buffer_type(dev), c, as_data, b_data, rounds);
        auto got = run(backend, pbuft, c, as_data, b_data, rounds);

        size_t mismatches = 0;
        double max_diff   = 0;
        for (size_t r = 0; r < ref.size(); r++) {
            for (size_t i = 0; i < ref[r].size(); i++) {
                if (memcmp(&ref[r][i], &got[r][i], sizeof(float)) != 0) {
                    mismatches++;
                    max_diff = std::max(max_diff, (double) std::fabs(ref[r][i] - got[r][i]));
                }
            }
        }
        const bool ok = mismatches == 0 && (!c.source || (store.reads > 0 && store.batches > 0));
        failed += ok ? 0 : 1;
        printf("  %s %-5s experts=%3d used=%d tokens=%2d slots=%3d %s: %s (mismatches %zu, max diff %g, reads %d in %d batches)\n",
               ok ? "OK  " : "FAIL", ggml_type_name(c.type), c.n_expert, c.n_used, c.n_tokens, c.n_slots,
               c.source ? "source" : "ram   ", ok ? "identical" : "differs", mismatches, max_diff, store.reads, store.batches);
    }
    printf("%zu/%zu configs identical\n", configs.size() - failed, configs.size());
    ggml_backend_free(backend);
    return failed == 0 ? 0 : 1;
}
