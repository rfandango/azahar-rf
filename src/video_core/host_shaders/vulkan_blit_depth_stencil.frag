// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#version 450 core
#extension GL_ARB_shader_stencil_export : require

layout(binding = 0) uniform sampler2D depth_tex;
#ifdef STENCIL_BYTES
// The stencil, copied into a buffer a byte a texel of depth_tex (an image whose stencil is
// sampled could not be compressed on every GPU)
layout(binding = 2) uniform usamplerBuffer stencil_bytes;
#else
layout(binding = 1) uniform usampler2D stencil_tex;
#endif

layout(push_constant) uniform PushConstants {
    vec2 tex_scale;
    vec2 tex_offset;
    ivec2 stencil_at; // first byte of the stencil copy, its row length
};

layout(location = 0) in vec2 texcoord;

void main() {
    gl_FragDepth = textureLod(depth_tex, texcoord, 0).r;
#ifdef STENCIL_BYTES
    // The texel the nearest sample above takes
    ivec2 size = textureSize(depth_tex, 0);
    ivec2 texel = clamp(ivec2(floor(texcoord * vec2(size))), ivec2(0), size - 1);
    gl_FragStencilRefARB =
        int(texelFetch(stencil_bytes, stencil_at.x + texel.y * stencil_at.y + texel.x).r);
#else
    gl_FragStencilRefARB = int(textureLod(stencil_tex, texcoord, 0).r);
#endif
}
