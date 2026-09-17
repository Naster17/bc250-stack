// RPP add_scalar gate: 2-image f32 batch on HIP backend vs CPU reference.
// Layout NCHW contiguous, full-tensor ROI, scalar add per image.
#include <cmath>
#include <cstdio>
#include <vector>

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppt_tensor_arithmetic_operations.h>

int main(int argc, char** argv) {
    const bool host_only = (argc > 1);
    const unsigned batch = 2, c = 3, h = 16, w = 16;
    const size_t elems = static_cast<size_t>(batch) * c * h * w;
    std::vector<float> src(elems);
    for (size_t i = 0; i < elems; ++i) {
        src[i] = static_cast<float>((i * 7) % 251) / 255.0f;
    }
    const float adds[2] = {0.25f, -0.125f};
    std::vector<float> ref(elems);
    for (unsigned n = 0; n < batch; ++n) {
        for (size_t i = 0; i < static_cast<size_t>(c) * h * w; ++i) {
            ref[n * c * h * w + i] = src[n * c * h * w + i] + adds[n];
        }
    }
    float* d_src = nullptr;
    float* d_dst = nullptr;
    float* d_add = nullptr;
    if (hipMalloc(&d_src, elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_dst, elems * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_add, batch * sizeof(float)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed\n");
        return 1;
    }
    hipMemcpy(d_src, src.data(), elems * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_add, adds, batch * sizeof(float), hipMemcpyHostToDevice);
    RpptGenericDesc desc = {};
    desc.numDims = 5;
    desc.offsetInBytes = 0;
    desc.dataType = F32;
    desc.dims[0] = batch;
    desc.dims[1] = c;
    desc.dims[2] = 1;
    desc.dims[3] = h;
    desc.dims[4] = w;
    desc.strides[0] = c * h * w;
    desc.strides[1] = h * w;
    desc.strides[2] = h * w;
    desc.strides[3] = w;
    desc.strides[4] = 1;
    desc.layout = NCDHW;
    // NOTE: RPP HIP kernels dereference the ROI pointer on-device, so it must be
    // device-visible. Host-stack memory faults with xnack-; use mapped pinned memory.
    RpptROI3D* h_rois = nullptr;
    RpptROI3D* d_rois = nullptr;
    if (hipHostMalloc(&h_rois, batch * sizeof(RpptROI3D), hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer(reinterpret_cast<void**>(&d_rois), h_rois, 0) != hipSuccess) {
        std::fprintf(stderr, "mapped ROI alloc failed\n");
        return 1;
    }
    for (unsigned n = 0; n < batch; ++n) {
        h_rois[n].xyzwhdROI.xyz.x = 0;
        h_rois[n].xyzwhdROI.xyz.y = 0;
        h_rois[n].xyzwhdROI.xyz.z = 0;
        h_rois[n].xyzwhdROI.roiWidth = w;
        h_rois[n].xyzwhdROI.roiHeight = h;
        h_rois[n].xyzwhdROI.roiDepth = 1;
    }
    rppHandle_t handle = nullptr;
    if (rppCreate(&handle, batch, 0, nullptr, host_only ? RPP_HOST_BACKEND : RPP_HIP_BACKEND) != RPP_SUCCESS) {
        std::fprintf(stderr, "rppCreate failed\n");
        return 1;
    }
    std::vector<float> dst(elems, 0.0f);
    RppStatus status;
    if (host_only) {
        status = rppt_add_scalar(src.data(), &desc, dst.data(), &desc, const_cast<float*>(adds),
                                 h_rois, XYZWHD, handle, RPP_HOST_BACKEND);
    } else {
        status = rppt_add_scalar(d_src, &desc, d_dst, &desc, d_add, d_rois, XYZWHD, handle, RPP_HIP_BACKEND);
    }
    std::printf("RPP_STATUS=%d\n", static_cast<int>(status));
    bool ok = (status == RPP_SUCCESS);
    if (ok) {
        hipError_t serr = hipDeviceSynchronize();
        std::printf("RPP_SYNC=%d\n", static_cast<int>(serr));
        ok = (serr == hipSuccess);
    }
    if (ok && !host_only) {
        ok = (hipMemcpy(dst.data(), d_dst, elems * sizeof(float), hipMemcpyDeviceToHost) ==
              hipSuccess);
    }
    for (size_t i = 0; ok && i < elems; ++i) {
        if (std::fabs(dst[i] - ref[i]) > 1.0e-5) {
            std::fprintf(stderr, "output %zu mismatch: got %g, expected %g\n", i, dst[i], ref[i]);
            ok = false;
        }
    }
    if (handle) {
        rppDestroy(handle);
    }
    hipFree(d_src);
    hipFree(d_dst);
    hipFree(d_add);
    hipHostFree(h_rois);
    std::printf("RPP_ADD_SCALAR_VERIFY_%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
