/// @file errors.hpp
/// @brief Overlay-specific errors.

#pragma once

#include "jarvis/common/errors.hpp"

namespace jarvis::overlay {

/// The overlay window, its Direct3D device or its swap chain could not be
/// created. Typical causes: no GPU and no WARP software rasteriser (rare),
/// or a session without a desktop (service, locked CI runner).
class OverlayError : public NativeError {
public:
    using NativeError::NativeError;
};

}  // namespace jarvis::overlay
