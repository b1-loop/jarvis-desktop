/// @file onnx_session.hpp
/// @brief Minimal RAII wrapper over an ONNX Runtime inference session.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace jarvis::audio {

/// Loads one ONNX model and runs it on the CPU.
///
/// Exists so that the VAD and wake-word classes read like signal processing
/// rather than ONNX Runtime boilerplate, and so the heavy ONNX headers stay
/// inside one translation unit. Input and output names are discovered from
/// the model, which lets callers address tensors by name instead of relying
/// on a fixed order.
class OnnxSession {
public:
    /// A float32 input tensor. The span must outlive the run() call.
    struct FloatInput {
        std::span<const float> data;
        std::vector<std::int64_t> shape;
    };

    /// An int64 input tensor (used for scalar parameters such as sample rate).
    struct Int64Input {
        std::span<const std::int64_t> data;
        std::vector<std::int64_t> shape;
    };

    using Input = std::variant<FloatInput, Int64Input>;

    /// A float32 output tensor, copied out of the runtime.
    struct Tensor {
        std::vector<float> data;
        std::vector<std::int64_t> shape;
    };

    /// Load @p model. Throws ModelError if the file is missing or invalid.
    explicit OnnxSession(const std::filesystem::path& model);
    ~OnnxSession();

    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;

    [[nodiscard]] std::size_t inputCount() const noexcept;

    /// Index of the input called @p name. Throws ModelError if absent.
    [[nodiscard]] std::size_t inputIndex(std::string_view name) const;

    /// Index of the output called @p name. Throws ModelError if absent.
    [[nodiscard]] std::size_t outputIndex(std::string_view name) const;

    /// Run the model. @p inputs must be in model input order, one per input.
    /// @return Every output, in model output order.
    [[nodiscard]] std::vector<Tensor> run(std::span<const Input> inputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::audio
