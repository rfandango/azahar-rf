// Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Snapdragon Game Super Resolution 1, with EDGE_DIRECTION the variant of that name, from
// https://github.com/SnapdragonGameStudios/snapdragon-gsr (sgsr/v1), whose license is in
// misc/licenses/snapdragon-gsr.txt. Changed for Azahar: it draws a screen of the console, which may be a
// part of a larger image.

#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in highp vec2 frag_tex_coord;
layout (location = 0) out vec4 color_out;

layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    // The part of the image that is the screen: its lowest and highest coordinates
    vec4 screen_area;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
    int reverse_interlaced;
    // 0 to 2
    float edge_sharpness;
};

layout (set = 0, binding = 0) uniform sampler2D screen_textures[3];

// What the picture's edges are looked for in: green
#define GATHER_COMPONENT 1
#define EDGE_THRESHOLD (8.0 / 255.0)

vec4 SampleScreen(highp vec2 coord) {
#ifdef ARRAY_DYNAMIC_INDEX
    return textureLod(screen_textures[screen_id_l], coord, 0.0);
#else
    switch (screen_id_l) {
    case 0:
        return textureLod(screen_textures[0], coord, 0.0);
    case 1:
        return textureLod(screen_textures[1], coord, 0.0);
    default:
        return textureLod(screen_textures[2], coord, 0.0);
    }
#endif
}

vec4 GatherScreen(highp vec2 coord) {
#ifdef ARRAY_DYNAMIC_INDEX
    return textureGather(screen_textures[screen_id_l], coord, GATHER_COMPONENT);
#else
    switch (screen_id_l) {
    case 0:
        return textureGather(screen_textures[0], coord, GATHER_COMPONENT);
    case 1:
        return textureGather(screen_textures[1], coord, GATHER_COMPONENT);
    default:
        return textureGather(screen_textures[2], coord, GATHER_COMPONENT);
    }
#endif
}

highp vec2 ScreenSize() {
#ifdef ARRAY_DYNAMIC_INDEX
    return vec2(textureSize(screen_textures[screen_id_l], 0));
#else
    switch (screen_id_l) {
    case 0:
        return vec2(textureSize(screen_textures[0], 0));
    case 1:
        return vec2(textureSize(screen_textures[1], 0));
    default:
        return vec2(textureSize(screen_textures[2], 0));
    }
#endif
}

mediump float FastLanczos2(mediump float x) {
    mediump float wA = x - 4.0;
    mediump float wB = x * wA - wA;
    wA *= wA;
    return wB * wA;
}

// (`data` holds a value that goes beyond what half precision can, at edges that are just above the
// threshold)
mediump vec2 WeightY(mediump float dx, mediump float dy, mediump float c, highp vec3 data) {
    highp float std = data.x;
#ifdef EDGE_DIRECTION
    mediump vec2 dir = data.yz;

    mediump float edge_dis = (dx * dir.y) + (dy * dir.x);
    mediump float x = ((dx * dx) + (dy * dy)) +
                      ((edge_dis * edge_dis) * ((clamp((c * c) * std, 0.0, 1.0) * 0.7) + -1.0));
#else
    mediump float x = ((dx * dx) + (dy * dy)) * 0.55 + clamp(abs(c) * std, 0.0, 1.0);
#endif
    mediump float w = FastLanczos2(x);
    return vec2(w, w * c);
}

mediump vec2 EdgeDirection(mediump vec4 left, mediump vec4 right) {
    mediump float RxLz = right.x + (-left.z);
    mediump float RwLy = right.w + (-left.y);
    highp vec2 delta = vec2(RxLz + RwLy, RxLz + (-RwLy));
    highp float length_inv =
        inversesqrt((delta.x * delta.x + 3.075740e-05) + (delta.y * delta.y));
    return delta * length_inv;
}

void main() {
    mediump vec4 color = vec4(SampleScreen(frag_tex_coord).rgb, 1.0);

    highp vec2 size = ScreenSize();
    highp vec2 texel = 1.0 / size;
    highp vec2 img_coord = (frag_tex_coord * size) + vec2(-0.5, 0.5);
    highp vec2 img_coord_pixel = floor(img_coord);
    highp vec2 coord = img_coord_pixel * texel;
    mediump vec2 pl = img_coord - img_coord_pixel;

    // Four columns and rows are looked at, from one before to two after. Where they are not all the
    // screen's the picture stays as it is.
    highp vec2 first = coord + vec2(-1.0, -2.0) * texel;
    highp vec2 last = coord + vec2(3.0, 2.0) * texel;
    if (any(lessThan(first, screen_area.xy - 0.01 * texel)) ||
        any(greaterThan(last, screen_area.zw + 0.01 * texel))) {
        color_out = color;
        return;
    }

    mediump vec4 left = GatherScreen(coord);
    mediump float center = color[GATHER_COMPONENT];
    mediump float edge_vote =
        abs(left.z - left.y) + abs(center - left.y) + abs(center - left.z);
    if (edge_vote > EDGE_THRESHOLD) {
        coord.x += texel.x;

        mediump vec4 right = GatherScreen(coord + vec2(texel.x, 0.0));
        mediump vec4 up_down;
        up_down.xy = GatherScreen(coord + vec2(0.0, -texel.y)).wz;
        up_down.zw = GatherScreen(coord + vec2(0.0, texel.y)).yx;

        mediump float mean = (left.y + left.z + right.x + right.w) * 0.25;
        left -= vec4(mean);
        right -= vec4(mean);
        up_down -= vec4(mean);
        center -= mean;

        mediump float sum = dot(abs(left), vec4(1.0)) + dot(abs(right), vec4(1.0)) +
                            dot(abs(up_down), vec4(1.0));
#ifdef EDGE_DIRECTION
        highp float sum_mean = 1.014185e+01 / sum;
        highp vec3 data = vec3(sum_mean * sum_mean, EdgeDirection(left, right));
#else
        highp vec3 data = vec3(2.181818 / sum, 0.0, 0.0);
#endif

        mediump vec2 aWY = WeightY(pl.x, pl.y + 1.0, up_down.x, data);
        aWY += WeightY(pl.x - 1.0, pl.y + 1.0, up_down.y, data);
        aWY += WeightY(pl.x - 1.0, pl.y - 2.0, up_down.z, data);
        aWY += WeightY(pl.x, pl.y - 2.0, up_down.w, data);
        aWY += WeightY(pl.x + 1.0, pl.y - 1.0, left.x, data);
        aWY += WeightY(pl.x, pl.y - 1.0, left.y, data);
        aWY += WeightY(pl.x, pl.y, left.z, data);
        aWY += WeightY(pl.x + 1.0, pl.y, left.w, data);
        aWY += WeightY(pl.x - 1.0, pl.y - 1.0, right.x, data);
        aWY += WeightY(pl.x - 2.0, pl.y - 1.0, right.y, data);
        aWY += WeightY(pl.x - 2.0, pl.y, right.z, data);
        aWY += WeightY(pl.x - 1.0, pl.y, right.w, data);

        mediump float final_y = aWY.y / aWY.x;
        mediump float max_y = max(max(left.y, left.z), max(right.x, right.w));
        mediump float min_y = min(min(left.y, left.z), min(right.x, right.w));
        mediump float delta_y = clamp(edge_sharpness * final_y, min_y, max_y) - center;

        // Smooth high contrast input
        delta_y = clamp(delta_y, -23.0 / 255.0, 23.0 / 255.0);
        color.rgb = clamp(color.rgb + vec3(delta_y), 0.0, 1.0);
    }

    color_out = color;
}
