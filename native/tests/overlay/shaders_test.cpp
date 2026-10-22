#include "jarvis/overlay/shaders.hpp"

#include <catch2/catch_test_macros.hpp>

#include <d3dcompiler.h>
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

using jarvis::overlay::ShaderId;
using jarvis::overlay::ShaderSource;

namespace {

struct LibraryCloser {
    void operator()(HMODULE module) const noexcept {
        if (module != nullptr) {
            FreeLibrary(module);
        }
    }
};
using UniqueLibrary = std::unique_ptr<std::remove_pointer_t<HMODULE>, LibraryCloser>;

/// Releases a COM blob.
struct BlobReleaser {
    void operator()(ID3DBlob* blob) const noexcept {
        if (blob != nullptr) {
            blob->Release();
        }
    }
};
using UniqueBlob = std::unique_ptr<ID3DBlob, BlobReleaser>;

/// Compile @p shader; returns the compiler's message, empty on success.
std::string compileMessage(pD3DCompile compile, const ShaderSource& shader) {
    const std::string name{shader.name};
    const std::string entry{shader.entry};
    const std::string profile{shader.profile};
    ID3DBlob* bytecode = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT result = compile(shader.code.data(), shader.code.size(), name.c_str(), nullptr,
                                   nullptr, entry.c_str(), profile.c_str(),
                                   D3DCOMPILE_ENABLE_STRICTNESS, 0, &bytecode, &errors);
    const UniqueBlob ownedBytecode{bytecode};
    const UniqueBlob ownedErrors{errors};
    if (SUCCEEDED(result) && ownedBytecode != nullptr && ownedBytecode->GetBufferSize() > 0) {
        return {};
    }
    std::string message = name + " failed";
    if (ownedErrors != nullptr) {
        message.append(": ").append(static_cast<const char*>(ownedErrors->GetBufferPointer()),
                                    ownedErrors->GetBufferSize());
    }
    return message;
}

/// True when the shaders are in ShaderId order, unique, findable by id and
/// carry the shared prelude and an entry point.
bool wellFormed(std::span<const ShaderSource> shaders) {
    std::set<std::string_view> names;
    std::size_t index = 0;
    return std::ranges::all_of(shaders, [&names, &index](const ShaderSource& shader) {
        return static_cast<std::size_t>(shader.id) == index++ &&
               &jarvis::overlay::shaderSource(shader.id) == &shader &&
               names.insert(shader.name).second &&
               shader.code.find("cbuffer PostParams") != std::string_view::npos &&
               shader.code.find("main(") != std::string_view::npos;
    });
}

}  // namespace

TEST_CASE("every post-processing pass has a unique name and the shared prelude",
          "[overlay][shaders]") {
    const auto shaders = jarvis::overlay::shaderSources();
    REQUIRE(shaders.size() == 5);
    REQUIRE(wellFormed(shaders));
    REQUIRE(jarvis::overlay::shaderSource(ShaderId::Fullscreen).profile == "vs_4_0");
    REQUIRE(jarvis::overlay::shaderSource(ShaderId::Composite).profile == "ps_4_0");
}

TEST_CASE("every post-processing shader compiles with D3DCompile", "[overlay][shaders]") {
    const UniqueLibrary library{LoadLibraryW(L"d3dcompiler_47.dll")};
    if (library == nullptr) {
        SKIP("d3dcompiler_47.dll is not available on this machine");
    }
    // GetProcAddress returns a generic function pointer by design.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto compile = reinterpret_cast<pD3DCompile>(GetProcAddress(library.get(), "D3DCompile"));
    if (compile == nullptr) {
        SKIP("d3dcompiler_47.dll has no D3DCompile export");
    }
    for (const ShaderSource& shader : jarvis::overlay::shaderSources()) {
        INFO("shader " << shader.name);
        CHECK(compileMessage(compile, shader).empty());
        INFO(compileMessage(compile, shader));
    }
}
