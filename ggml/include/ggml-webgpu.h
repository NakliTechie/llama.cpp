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

// Paged expert weights for MUL_MAT_ID: the GPU keeps n_slots experts per tensor. Before each MUL_MAT_ID the backend
// reads back the router ids and copies missing experts in from the page source, evicting the least recently used.
struct ggml_webgpu_page_source {
    void * user_data;
    // optional: true when the source already holds the tensor, so set_tensor skips write
    bool (*has)(void * user_data, const char * tensor_name, size_t nbytes);
    // model load: bytes [offset, offset + size) of the tensor
    void (*write)(void * user_data, const char * tensor_name, size_t offset, const void * data, size_t size);
    // compute: bytes [offset, offset + size) of the tensor (one expert)
    void (*read)(void * user_data, const char * tensor_name, size_t offset, void * data, size_t size);
    // optional: n reads in one call, which the source may run concurrently
    void (*read_batch)(void *               user_data,
                       size_t               n,
                       const char * const * tensor_names,
                       const size_t *       offsets,
                       void * const *       data,
                       const size_t *       sizes);
};

// source NULL keeps the expert bytes in host memory. The returned buffer type is never freed.
GGML_BACKEND_API ggml_backend_buffer_type_t
ggml_backend_webgpu_paged_buffer_type(ggml_backend_dev_t                     dev,
                                      uint32_t                               n_slots,
                                      const struct ggml_webgpu_page_source * source);

#ifdef  __cplusplus
}
#endif
