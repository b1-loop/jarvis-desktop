#include "jarvis/overlay/effects.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>

using Catch::Matchers::WithinAbs;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr float kTolerance = 1e-5F;
constexpr std::array<anim::Intensity, 3> kOrder{anim::Intensity::Calm, anim::Intensity::Normal,
                                                anim::Intensity::Max};

constexpr int kGlitchSamples = 2000;

/// True when every field of the higher intensity's scale is at least the lower's.
bool grows(anim::Intensity lower, anim::Intensity higher) {
    const anim::IntensityScale a = anim::intensityScale(lower);
    const anim::IntensityScale b = anim::intensityScale(higher);
    return b.particles > a.particles && b.bloom > a.bloom && b.motion > a.motion &&
           b.bloomLevels >= a.bloomLevels &&
           anim::particleBudget(higher, 900) > anim::particleBudget(lower, 900);
}

struct GlitchSurvey {
    bool bounded{true};
    int quiet{0};
};

/// Samples the boot's glitch envelope: is it in [0, 1], and how often is it quiet?
GlitchSurvey surveyGlitch(const anim::BootTimeline& timeline) {
    GlitchSurvey survey;
    for (int i = 0; i <= kGlitchSamples; ++i) {
        const float time = timeline.duration() * static_cast<float>(i) / kGlitchSamples;
        const float glitch = anim::glitchAmount(timeline.spikes(), time);
        survey.bounded = survey.bounded && glitch >= 0.0F && glitch <= 1.0F;
        survey.quiet += glitch < 0.05F ? 1 : 0;
    }
    return survey;
}

/// True when bloom stays in [0, 2] for every combination.
bool bloomBounded(std::span<const anim::StateWeights> weights,
                  std::span<const anim::AudioEnergy> energies, float breath) {
    return std::ranges::all_of(weights, [&](const anim::StateWeights& w) {
        return std::ranges::all_of(energies, [&](const anim::AudioEnergy& e) {
            const float bloom = anim::bloomStrength(w, e, breath);
            return bloom >= 0.0F && bloom <= 2.0F;
        });
    });
}

}  // namespace

// --- intensity ---------------------------------------------------------------------

TEST_CASE("intensity names round-trip and unknown names are rejected", "[overlay][effects]") {
    REQUIRE(std::ranges::all_of(kOrder, [](anim::Intensity intensity) {
        return anim::parseIntensity(anim::toString(intensity)) == intensity;
    }));
    REQUIRE(anim::toString(anim::Intensity::Normal) == "normal");
    REQUIRE_FALSE(anim::parseIntensity("loud").has_value());
    REQUIRE_FALSE(anim::parseIntensity("Normal").has_value());
}

TEST_CASE("intensity scaling is monotonic in every field", "[overlay][effects]") {
    REQUIRE(grows(anim::Intensity::Calm, anim::Intensity::Normal));
    REQUIRE(grows(anim::Intensity::Normal, anim::Intensity::Max));
    REQUIRE(std::ranges::all_of(kOrder, [](anim::Intensity intensity) {
        const int levels = anim::intensityScale(intensity).bloomLevels;
        return levels >= 2 && levels <= 4;
    }));
}

TEST_CASE("normal intensity is the identity", "[overlay][effects]") {
    const anim::IntensityScale normal = anim::intensityScale(anim::Intensity::Normal);
    REQUIRE(normal.particles == 1.0F);
    REQUIRE(normal.bloom == 1.0F);
    REQUIRE(normal.motion == 1.0F);
    REQUIRE(anim::particleBudget(anim::Intensity::Normal, 900) == 900);
}

// --- glitch -------------------------------------------------------------------------

TEST_CASE("a glitch spike rises, peaks and decays", "[overlay][effects][glitch]") {
    const anim::GlitchSpike spike{.at = 1.0F, .strength = 0.8F, .decay = 0.1F};
    REQUIRE(anim::spikeEnvelope(spike, 0.99F) == 0.0F);
    REQUIRE(anim::spikeEnvelope(spike, 1.0F) == 0.0F);
    REQUIRE_THAT(anim::spikeEnvelope(spike, 1.015F), WithinAbs(0.8, 1e-3));
    REQUIRE(anim::spikeEnvelope(spike, 1.1F) < anim::spikeEnvelope(spike, 1.05F));
    REQUIRE(anim::spikeEnvelope(spike, 2.0F) < 1e-3F);
    REQUIRE(anim::spikeEnvelope({.at = 0.0F, .strength = 0.0F, .decay = 0.1F}, 0.01F) == 0.0F);
}

TEST_CASE("the glitch envelope is bounded and 0 most of the time", "[overlay][effects][glitch]") {
    const anim::BootTimeline timeline;
    const auto spikes = timeline.spikes();
    REQUIRE(spikes.size() > 5);
    const GlitchSurvey survey = surveyGlitch(timeline);
    REQUIRE(survey.bounded);
    REQUIRE(survey.quiet > kGlitchSamples / 3);  // spikes, not a constant shimmer
}

TEST_CASE("the glitch saturates at 1 and is 0 without spikes", "[overlay][effects][glitch]") {
    const std::array<anim::GlitchSpike, 2> loud{{{.at = 0.0F, .strength = 3.0F, .decay = 1.0F},
                                                 {.at = 0.0F, .strength = 2.0F, .decay = 1.0F}}};
    REQUIRE(anim::glitchAmount(loud, 0.1F) == 1.0F);
    REQUIRE(anim::glitchAmount({}, 0.1F) == 0.0F);
}

TEST_CASE("the wake glitch is a short hit", "[overlay][effects][glitch]") {
    REQUIRE(anim::wakeGlitch(0.0F) == 0.0F);
    REQUIRE(anim::wakeGlitch(1.0F) > 0.5F);
    REQUIRE(anim::wakeGlitch(1.0F) <= 1.0F);
    REQUIRE(anim::wakeGlitch(0.5F) < anim::wakeGlitch(1.0F) / 4.0F);
    REQUIRE(anim::wakeGlitch(7.0F) == anim::wakeGlitch(1.0F));
}

// --- audio reactivity --------------------------------------------------------------------

TEST_CASE("audio energy attacks fast, releases slowly and stays bounded", "[overlay][effects]") {
    anim::AudioReactor reactor;
    const anim::AudioEnergy& up = reactor.update(0.25F, 0.05F);
    REQUIRE(up.level > 0.4F);  // sqrt(0.25) = 0.5, reached within a few attack constants
    REQUIRE(up.peak == 0.5F);
    REQUIRE(up.onset > 0.0F);
    const float peak = reactor.energy().peak;
    reactor.update(0.0F, 0.1F);
    REQUIRE(reactor.energy().peak < peak);
    REQUIRE(reactor.energy().peak > reactor.energy().level);  // peak holds longer
}

TEST_CASE("audio energy stays bounded and resets", "[overlay][effects]") {
    anim::AudioReactor reactor;
    const auto bounded = [&reactor](float rms) {
        const anim::AudioEnergy& e = reactor.update(rms, 0.02F);
        return e.level >= 0.0F && e.level <= 1.0F && e.onset >= 0.0F && e.onset <= 1.0F;
    };
    REQUIRE(std::ranges::all_of(std::array{5.0F, -1.0F, 0.9F, 0.0F}, bounded));
    reactor.reset();
    REQUIRE(reactor.energy().level == 0.0F);
}

TEST_CASE("a steady tone has no onset but a new syllable has", "[overlay][effects]") {
    anim::AudioReactor reactor;
    for (int i = 0; i < 400; ++i) {
        reactor.update(0.16F, 0.01F);
    }
    REQUIRE(reactor.energy().onset < 0.01F);
    reactor.update(0.81F, 0.03F);
    REQUIRE(reactor.energy().onset > 0.3F);
}

TEST_CASE("audio smoothing depends on elapsed time, not frame count", "[overlay][effects]") {
    anim::AudioReactor once;
    anim::AudioReactor many;
    once.update(0.5F, 0.2F);
    for (int i = 0; i < 12; ++i) {
        many.update(0.5F, 0.2F / 12.0F);
    }
    REQUIRE_THAT(once.energy().level, WithinAbs(many.energy().level, 1e-4));
}

// --- per-frame looks --------------------------------------------------------------------

TEST_CASE("bloom is dim when idle and brightens with speech, within bounds", "[overlay][effects]") {
    const anim::StateWeights idle{1.0F, 0.0F, 0.0F, 0.0F};
    const anim::StateWeights speaking{0.0F, 0.0F, 0.0F, 1.0F};
    const anim::AudioEnergy quiet{};
    const anim::AudioEnergy loud{.level = 1.0F, .peak = 1.0F, .onset = 1.0F};
    REQUIRE(anim::bloomStrength(idle, quiet, 1.0F) < anim::bloomStrength(speaking, quiet, 0.0F));
    REQUIRE(anim::bloomStrength(speaking, loud, 0.0F) > anim::bloomStrength(speaking, quiet, 0.0F));
    const float breath = GENERATE(0.0F, 0.5F, 1.0F);
    REQUIRE(bloomBounded(std::array{idle, speaking}, std::array{quiet, loud}, breath));
}

TEST_CASE("particle forces follow the state and the level", "[overlay][effects]") {
    const anim::AudioEnergy loud{.level = 0.8F, .peak = 0.9F, .onset = 0.5F};
    const anim::FieldForces listening = anim::fieldForces({0, 1, 0, 0}, loud, 0.0F, 1.0F);
    const anim::FieldForces speaking = anim::fieldForces({0, 0, 0, 1}, loud, 0.0F, 1.0F);
    const anim::FieldForces thinking = anim::fieldForces({0, 0, 1, 0}, loud, 0.0F, 1.0F);
    const anim::FieldForces idle = anim::fieldForces({1, 0, 0, 0}, {}, 0.7F, 1.0F);
    REQUIRE(listening.pull > 0.5F);
    REQUIRE(listening.burst == 0.0F);
    REQUIRE(speaking.burst > 0.5F);
    REQUIRE(speaking.pull == 0.0F);
    REQUIRE(thinking.swirl == 1.0F);
    REQUIRE(idle.pull == 0.0F);
    REQUIRE(idle.burst == 0.0F);
    REQUIRE_THAT(idle.breathe, WithinAbs(0.7, kTolerance));
    REQUIRE(idle.motion < listening.motion);  // idle is calmer
}
