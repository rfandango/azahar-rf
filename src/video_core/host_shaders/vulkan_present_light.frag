// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// A screen enlarged with linear filtering and sharpened: what a pixel differs from its surroundings is
// made more of. A sixth of what Snapdragon Game Super Resolution costs.

#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in highp vec2 frag_tex_coord;
layout (location = 0) out mediump vec4 color_out;

layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    // The part of the image that is the screen: its lowest and highest coordinates
    vec4 screen_area;
    // Its size in pixels, and that of a pixel
    vec4 image_size;
    int screen_id_l;
    int screen_id_r;
    float output_rows;
    int reverse_interlaced;
    // 0 to 2
    float sharpness;
};

layout (set = 0, binding = 0) uniform mediump sampler2D screen_textures[3];

mediump vec3 SampleScreen(highp vec2 coord) {
#ifdef ARRAY_DYNAMIC_INDEX
    return textureLod(screen_textures[screen_id_l], coord, 0.0).rgb;
#else
    switch (screen_id_l) {
    case 0:
        return textureLod(screen_textures[0], coord, 0.0).rgb;
    case 1:
        return textureLod(screen_textures[1], coord, 0.0).rgb;
    default:
        return textureLod(screen_textures[2], coord, 0.0).rgb;
    }
#endif
}

void main() {
    highp vec2 across = vec2(image_size.z, 0.0);
    highp vec2 along = vec2(0.0, image_size.w);
    mediump vec3 color = SampleScreen(frag_tex_coord);
    mediump vec3 around = SampleScreen(frag_tex_coord - across) +
                          SampleScreen(frag_tex_coord + across) +
                          SampleScreen(frag_tex_coord - along) +
                          SampleScreen(frag_tex_coord + along);
    // (Within bounds, as a hard edge would get a border)
    mediump vec3 delta =
        clamp((color - 0.25 * around) * (0.5 * sharpness), -31.0 / 255.0, 31.0 / 255.0);
    color_out = vec4(color + delta, 1.0);
}
