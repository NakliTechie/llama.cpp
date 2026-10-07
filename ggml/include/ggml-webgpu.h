#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_WEBGPU_NAME "WebGPU"

// Needed for examples in ggml
GGML_BACKEND_API ggml_backend_t ggml_backend_webgpu_init(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_webgpu_reg(void);

// Paged expert tensors: a buffer type for the 3D expert weights of MUL_MAT_ID (`ffn_*_exps`). The GPU keeps only
// n_slots experts per tensor; before each MUL_MAT_ID the backend reads the router's ids and pages the chosen experts in
// from a page source, evicting the least recently used. Select it per tensor with llama's tensor_buft_overrides
// (`-ot exps=WebGPU_Paged`). Bytes reach the source once, through set_tensor at model load.
struct ggml_webgpu_page_source {
    void * user_data;
    // true when the source already holds this tensor's bytes, so set_tensor at load can drop them
    bool (*has)(void * user_data, const char * tensor_name, size_t nbytes);
    // load time: bytes [offset, offset + size) of the tensor
    void (*write)(void * user_data, const char * tensor_name, size_t offset, const void * data, size_t size);
    // compute time: one expert's bytes, [offset, offset + size) of the tensor
    void (*read)(void * user_data, const char * tensor_name, size_t offset, void * data, size_t size);
    // optional: n such reads at once, which the source may run concurrently (NULL: `read` is called n times)
    void (*read_batch)(void * user_data, size_t n, const char * const * tensor_names, const size_t * offsets,
                       void * const * data, const size_t * sizes);
    // optional: put n such ranges straight into GPU buffers (WGPUBuffer at dst_offsets) on the device's queue, e.g. through
    // mapped staging buffers. Used instead of read/read_batch when set; must be queued before it returns.
    void (*upload_batch)(void * user_data, void * device, size_t n, const char * const * tensor_names,
                         const size_t * offsets, const size_t * sizes, void * const * dst_buffers, const size_t * dst_offsets);
};

// source NULL keeps the expert bytes in host memory (a RAM tier). The returned type lives as long as the process.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_webgpu_paged_buffer_type(ggml_backend_dev_t dev, uint32_t n_slots,
                                                                                const struct ggml_webgpu_page_source * source);

#ifdef  __cplusplus
}
#endif
