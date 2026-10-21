#include "jarvis/audio/onnx_session.hpp"

#include "jarvis/common/errors.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace jarvis::audio {

namespace {

/// One process-wide runtime environment; ONNX Runtime requires it to outlive
/// every session and creating several is wasteful.
Ort::Env& sharedEnvironment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "jarvis");
    return environment;
}

Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions options;
    // The models are tiny; one thread avoids contention with the audio
    // worker and keeps latency predictable.
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

std::size_t indexOf(const std::vector<std::string>& names, std::string_view wanted,
                    std::string_view kind, const std::string& model) {
    const auto found = std::ranges::find(names, wanted);
    if (found == names.end()) {
        throw ModelError("model " + model + " has no " + std::string{kind} + " named '" +
                         std::string{wanted} + "'");
    }
    return static_cast<std::size_t>(std::distance(names.begin(), found));
}

}  // namespace

struct OnnxSession::Impl {
    std::string modelName;
    Ort::Session session;
    Ort::MemoryInfo memory;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePointers;
    std::vector<const char*> outputNamePointers;

    explicit Impl(const std::filesystem::path& model)
        : modelName(model.filename().string()),
          session(sharedEnvironment(), model.c_str(), makeSessionOptions()),
          memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        const Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
            inputNames.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        }
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
            outputNames.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        }
        for (const auto& name : inputNames) {
            inputNamePointers.push_back(name.c_str());
        }
        for (const auto& name : outputNames) {
            outputNamePointers.push_back(name.c_str());
        }
    }

    /// Owned copies of the input data: ONNX Runtime wants mutable pointers
    /// and the inputs are small, so copying beats a const_cast.
    struct OwnedInput {
        std::vector<float> floats;
        std::vector<std::int64_t> ints;
    };

    Ort::Value makeValue(const Input& input, OwnedInput& owned) {
        return std::visit(
            [&](const auto& typed) -> Ort::Value {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, FloatInput>) {
                    owned.floats.assign(typed.data.begin(), typed.data.end());
                    return Ort::Value::CreateTensor<float>(memory, owned.floats.data(),
                                                           owned.floats.size(), typed.shape.data(),
                                                           typed.shape.size());
                } else {
                    owned.ints.assign(typed.data.begin(), typed.data.end());
                    return Ort::Value::CreateTensor<std::int64_t>(
                        memory, owned.ints.data(), owned.ints.size(), typed.shape.data(),
                        typed.shape.size());
                }
            },
            input);
    }

    static Tensor copyOut(const Ort::Value& value, const std::string& model) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw ModelError("model " + model + " produced a non-float output");
        }
        const std::span<const float> data{value.GetTensorData<float>(), info.GetElementCount()};
        return Tensor{.data = std::vector<float>(data.begin(), data.end()),
                      .shape = info.GetShape()};
    }
};

OnnxSession::OnnxSession(const std::filesystem::path& model) {
    if (!std::filesystem::exists(model)) {
        throw ModelError("model file not found: " + model.string());
    }
    try {
        impl_ = std::make_unique<Impl>(model);
    } catch (const Ort::Exception& error) {
        throw ModelError("failed to load " + model.string() + ": " + error.what());
    }
}

OnnxSession::~OnnxSession() = default;
OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

std::size_t OnnxSession::inputCount() const noexcept {
    return impl_->inputNames.size();
}

std::size_t OnnxSession::outputCount() const noexcept {
    return impl_->outputNames.size();
}

std::size_t OnnxSession::inputIndex(std::string_view name) const {
    return indexOf(impl_->inputNames, name, "input", impl_->modelName);
}

std::size_t OnnxSession::outputIndex(std::string_view name) const {
    return indexOf(impl_->outputNames, name, "output", impl_->modelName);
}

std::vector<OnnxSession::Tensor> OnnxSession::run(std::span<const Input> inputs) {
    if (inputs.size() != impl_->inputNames.size()) {
        throw ModelError("model " + impl_->modelName + " expects " +
                         std::to_string(impl_->inputNames.size()) + " inputs, got " +
                         std::to_string(inputs.size()));
    }
    std::vector<Impl::OwnedInput> owned(inputs.size());
    std::vector<Ort::Value> values;
    values.reserve(inputs.size());
    auto ownedSlot = owned.begin();
    for (const auto& input : inputs) {
        values.push_back(impl_->makeValue(input, *ownedSlot));
        ++ownedSlot;
    }
    try {
        const auto outputs = impl_->session.Run(
            Ort::RunOptions{nullptr}, impl_->inputNamePointers.data(), values.data(), values.size(),
            impl_->outputNamePointers.data(), impl_->outputNamePointers.size());
        std::vector<Tensor> result;
        result.reserve(outputs.size());
        for (const auto& output : outputs) {
            result.push_back(Impl::copyOut(output, impl_->modelName));
        }
        return result;
    } catch (const Ort::Exception& error) {
        throw ModelError("inference failed for " + impl_->modelName + ": " + error.what());
    }
}

}  // namespace jarvis::audio
