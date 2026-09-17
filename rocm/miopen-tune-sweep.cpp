// MIOpen gfx1013 tuning sweep: run Find over a grid of common NCHW fp32
// configs (fwd + bwd-data + bwd-weights). Results land in the user find-DB
// (MIOPEN_USER_DB_PATH), which becomes our gfx1013_20.HIP.fdb.txt.
#include <cstdio>
#include <vector>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

struct Cfg {
    int n, c, h, w, k, r;
};

int main() {
    const std::vector<Cfg> cfgs = {
        {1, 16, 16, 16, 16, 3}, {8, 32, 32, 32, 64, 3},  {16, 64, 56, 56, 128, 3},
        {8, 64, 28, 28, 128, 3}, {16, 128, 28, 28, 256, 3}, {16, 256, 14, 14, 512, 3},
        {1, 3, 224, 224, 64, 7}, {8, 64, 56, 56, 64, 1},  {1, 32, 28, 28, 32, 5},
        {4, 16, 32, 32, 32, 3},
    };
    miopenHandle_t handle = nullptr;
    if (miopenCreate(&handle) != miopenStatusSuccess) {
        std::fprintf(stderr, "miopenCreate failed\n");
        return 1;
    }
    int done = 0;
    for (const auto& c : cfgs) {
        const int out = c.h; // stride 1, pad 1, 3x3 (or same for 1x1/5x5/7x7 approx)
        const size_t x_n = static_cast<size_t>(c.n) * c.c * c.h * c.w;
        const size_t w_n = static_cast<size_t>(c.k) * c.c * c.r * c.r;
        const size_t y_n = static_cast<size_t>(c.n) * c.k * out * out;
        float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
        if (hipMalloc(&d_x, x_n * 4) != hipSuccess || hipMalloc(&d_w, w_n * 4) != hipSuccess ||
            hipMalloc(&d_y, y_n * 4) != hipSuccess) {
            std::fprintf(stderr, "hipMalloc failed\n");
            return 1;
        }
        miopenTensorDescriptor_t x_desc = nullptr, y_desc = nullptr, w_desc = nullptr;
        miopenConvolutionDescriptor_t conv = nullptr;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&y_desc);
        miopenCreateTensorDescriptor(&w_desc);
        miopenCreateConvolutionDescriptor(&conv);
        miopenSet4dTensorDescriptor(x_desc, miopenFloat, c.n, c.c, c.h, c.w);
        miopenSet4dTensorDescriptor(y_desc, miopenFloat, c.n, c.k, out, out);
        miopenSet4dTensorDescriptor(w_desc, miopenFloat, c.k, c.c, c.r, c.r);
        const int pad = c.r / 2;
        miopenInitConvolutionDescriptor(conv, miopenConvolution, pad, pad, 1, 1, 1, 1);
        size_t ws = 0;
        miopenConvolutionForwardGetWorkSpaceSize(handle, w_desc, x_desc, conv, y_desc, &ws);
        void* d_ws = nullptr;
        if (ws) {
            hipMalloc(&d_ws, ws);
        }
        miopenConvAlgoPerf_t perf;
        int ret = 0;
        miopenStatus_t st =
            miopenFindConvolutionForwardAlgorithm(handle, x_desc, d_x, w_desc, d_w, conv, y_desc,
                                                 d_y, 1, &ret, &perf, d_ws, ws, false);
        std::printf("TUNE fwd n=%d c=%d hw=%d k=%d r=%d -> %s algo=%d\n", c.n, c.c, c.h, c.k,
                    c.r, st == miopenStatusSuccess ? "OK" : "FAIL",
                    st == miopenStatusSuccess ? static_cast<int>(perf.fwd_algo) : -1);
        bool ok_cfg = (st == miopenStatusSuccess);
        miopenConvAlgoPerf_t bperf;
        int bret = 0;
        miopenStatus_t bst = miopenFindConvolutionBackwardDataAlgorithm(
            handle, y_desc, d_y, w_desc, d_w, conv, x_desc, d_x, 1, &bret, &bperf, d_ws, ws,
            false);
        std::printf("TUNE bwd-data n=%d c=%d hw=%d k=%d r=%d -> %s algo=%d\n", c.n, c.c, c.h,
                    c.k, c.r, bst == miopenStatusSuccess ? "OK" : "FAIL",
                    bst == miopenStatusSuccess ? static_cast<int>(bperf.bwd_data_algo) : -1);
        ok_cfg = ok_cfg && (bst == miopenStatusSuccess);
        miopenConvAlgoPerf_t wperf;
        int wret = 0;
        miopenStatus_t wst = miopenFindConvolutionBackwardWeightsAlgorithm(
            handle, y_desc, d_y, x_desc, d_x, conv, w_desc, d_w, 1, &wret, &wperf, d_ws, ws,
            false);
        std::printf("TUNE bwd-w n=%d c=%d hw=%d k=%d r=%d -> %s algo=%d\n", c.n, c.c, c.h, c.k,
                    c.r, wst == miopenStatusSuccess ? "OK" : "FAIL",
                    wst == miopenStatusSuccess ? static_cast<int>(wperf.bwd_weights_algo) : -1);
        ok_cfg = ok_cfg && (wst == miopenStatusSuccess);
        if (ok_cfg) {
            ++done;
        }
        miopenDestroyConvolutionDescriptor(conv);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        miopenDestroyTensorDescriptor(x_desc);
        hipFree(d_x);
        hipFree(d_w);
        hipFree(d_y);
        if (d_ws) {
            hipFree(d_ws);
        }
    }
    miopenDestroy(handle);
    std::printf("TUNE_DONE %d/%zu\n", done, cfgs.size());
    return done == static_cast<int>(cfgs.size()) ? 0 : 1;
}
