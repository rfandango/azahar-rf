// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include "video_core/pica/regs_lighting.h"

namespace Pica {

// (Not in regs_lighting.h, which the stored shaders are tied to the text of)

/// What a draw's fragments go by of the light at a place among those that are on. The rest of what
/// its registers hold was left there by other draws. (Monster Hunter has lighting on for the Fresnel
/// table alone, which it looks up by the view, with one light that is black: where that points to
/// nobody has set.)
struct UsedOfLight {
    /// Where it is or comes from, and its configuration
    bool vector;
    bool spot;
    bool distance;
};

inline UsedOfLight UsedOf(const LightingRegs& regs, unsigned place) {
    using Sampler = LightingRegs::LightingSampler;
    using Input = LightingRegs::LightingLutInput;
    const unsigned num = regs.light_enable.GetNum(place);
    const LightingRegs::LightSrc& source = regs.light[num];
    const auto& config1 = regs.config1;
    const auto is_black = [](const LightingRegs::LightColor& color) {
        return color.r == 0 && color.g == 0 && color.b == 0;
    };
    const bool black = is_black(source.specular_0) && is_black(source.specular_1) &&
                       is_black(source.diffuse) && is_black(source.ambient);
    const auto has = [&](Sampler sampler) {
        return LightingRegs::IsLightingSamplerSupported(regs.config0.config, sampler);
    };
    // The last light is also what the Fresnel table is looked up by, if by a light at all
    const bool fresnel =
        place == regs.max_light_index && config1.disable_lut_fr == 0 && has(Sampler::Fresnel) &&
        (regs.config0.enable_primary_alpha != 0 || regs.config0.enable_secondary_alpha != 0) &&
        regs.lut_input.fr != Input::NV;
    if (black && !fresnel) {
        return {};
    }
    const auto by_spot = [](bool disabled, Input input) {
        return !disabled && input == Input::SP;
    };
    const bool spot = (!regs.IsSpotAttenDisabled(num) && has(Sampler::SpotlightAttenuation)) ||
                      by_spot(config1.disable_lut_d0 != 0, regs.lut_input.d0) ||
                      by_spot(config1.disable_lut_d1 != 0, regs.lut_input.d1) ||
                      by_spot(config1.disable_lut_fr != 0, regs.lut_input.fr) ||
                      by_spot(config1.disable_lut_rr != 0, regs.lut_input.rr) ||
                      by_spot(config1.disable_lut_rg != 0, regs.lut_input.rg) ||
                      by_spot(config1.disable_lut_rb != 0, regs.lut_input.rb);
    return {.vector = true, .spot = spot, .distance = !regs.IsDistAttenDisabled(num)};
}

} // namespace Pica
