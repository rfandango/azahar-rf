// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// A screen enlarged with a cubic filter (Keys; at a sharpness of 1 it is a Catmull-Rom spline), by five
// samples that the hardware filters: the sixteen texels around are four columns and four rows,
// of which the two in the middle go into one sample and the corners, which weigh next to nothing,
// are left out. A third of what Snapdragon Game Super Resolution costs.

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
    highp vec2 position = frag_tex_coord * image_size.xy - 0.5;
    highp vec2 base = floor(position);
    mediump vec2 f = position - base;
    mediump vec2 g = 1.0 - f;
    mediump float a = -0.5 * sharpness;

    // The weights of one before, of the two in the middle and of one after
    mediump vec2 fg = f * g * a;
    mediump vec2 w0 = fg * g;
    mediump vec2 w3 = fg * f;
    mediump vec2 w12 = 1.0 - fg;
    mediump vec2 w2 = f * (f * ((2.0 * a + 3.0) - (a + 2.0) * f) - a);

    highp vec2 first = (base + 0.5) * image_size.zw;
    highp vec2 middle = first + (w2 / w12) * image_size.zw;
    highp vec2 before = first - image_size.zw;
    highp vec2 after = first + 2.0 * image_size.zw;

    mediump vec3 row = SampleScreen(middle) * w12.x +
                       SampleScreen(vec2(before.x, middle.y)) * w0.x +
                       SampleScreen(vec2(after.x, middle.y)) * w3.x;
    mediump vec3 column = SampleScreen(vec2(middle.x, before.y)) * w0.y +
                          SampleScreen(vec2(middle.x, after.y)) * w3.y;
    mediump float all = 1.0 - fg.x * fg.y;
    color_out = vec4((row * w12.y + column * w12.x) / all, 1.0);
}
