#include "backend/backend.h"
#include "backend/cpu_backend.h"
#ifdef ENABLE_QNN_STUB
#include "backend/qnn_stub_backend.h"
#endif

std::unique_ptr<Backend> create_backend(BackendType type) {
  if (type == BackendType::CPU)
    return std::make_unique<CPUBackend>();
#ifdef ENABLE_QNN_STUB
  if (type == BackendType::QNN)
    return std::make_unique<QnnStubBackend>();
#endif
  return std::make_unique<CPUBackend>();
}
