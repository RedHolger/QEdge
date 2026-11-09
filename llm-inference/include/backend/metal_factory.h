#pragma once
#include <memory>
#include "backend/backend.h"

std::unique_ptr<Backend> create_metal_backend();
