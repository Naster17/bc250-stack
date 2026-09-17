// hipDNN backend gate: library version + handle create/destroy round-trip.
#include <cstdio>

#include <hipdnn/backend/hipdnn_backend.h>

int main() {
    const char* version = nullptr;
    hipdnnStatus_t status = hipdnnGetVersion_ext(&version);
    bool ok = (status == HIPDNN_STATUS_SUCCESS && version != nullptr);
    std::printf("HIPDNN_VERSION %s\n", ok ? version : "unknown");
    hipdnnHandle_t handle = nullptr;
    if (ok) {
        status = hipdnnCreate(&handle);
        ok = (status == HIPDNN_STATUS_SUCCESS && handle != nullptr);
        std::printf("HIPDNN_CREATE %s\n", ok ? "OK" : "FAILED");
    }
    if (handle != nullptr) {
        status = hipdnnDestroy(handle);
        ok = (status == HIPDNN_STATUS_SUCCESS) && ok;
    }
    std::printf("HIPDNN_BACKEND_VERIFY_%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
