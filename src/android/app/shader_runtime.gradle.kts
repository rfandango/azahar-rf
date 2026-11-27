// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

// Puts librashader into the app, which then runs RetroArch shader presets (libretro's shader pack,
// downloaded in the app) with the Vulkan renderer. It is not part of the source: the property
// shaderRuntime names a folder with a folder for each ABI (arm64-v8a, x86_64), each holding
// librashader.so and the libc++_shared.so it needs, built from librashader's C API
// (https://github.com/SnowflakePowered/librashader, MPL-2.0 or GPL-3.0). Without one the app
// comes without it and does not offer the presets.

val runtimeFolder: File? = providers.gradleProperty("shaderRuntime").orNull?.trim()
    ?.takeIf { it.isNotEmpty() && it != "none" }
    ?.let { property ->
        val path = if (property.startsWith("~/")) {
            System.getProperty("user.home") + property.drop(1)
        } else {
            property
        }
        gradle.startParameter.currentDir.resolve(path).also {
            if (!it.isDirectory) {
                throw GradleException("shaderRuntime: there is no folder $it")
            }
        }
    }

val runtimeLibraries = listOf("librashader.so", "libc++_shared.so")

tasks.register("injectShaderRuntime") {
    val output = layout.buildDirectory.dir("generated/shaderRuntime").get().asFile
    inputs.property("folder", runtimeFolder?.path ?: "none")
    runtimeFolder?.let { inputs.dir(it) }
    outputs.dir(output)

    doLast {
        output.deleteRecursively()
        output.resolve("jniLibs").mkdirs()
        val source = runtimeFolder
        if (source == null) {
            logger.lifecycle("Shader preset runtime in the app: none")
            return@doLast
        }
        val abis = source.listFiles { f: File -> f.isDirectory }.orEmpty().map { it.name }.sorted()
        if (abis.isEmpty()) {
            throw GradleException("shaderRuntime $source: no folder for an ABI in it")
        }
        for (abi in abis) {
            val target = output.resolve("jniLibs/$abi").apply { mkdirs() }
            for (library in runtimeLibraries) {
                val file = source.resolve("$abi/$library")
                if (!file.isFile) {
                    throw GradleException("shaderRuntime $source: no $abi/$library")
                }
                file.copyTo(target.resolve(library), overwrite = true)
            }
        }
        logger.lifecycle("Shader preset runtime in the app: librashader for ${abis.joinToString()}")
    }
}

tasks.named("preBuild") {
    dependsOn("injectShaderRuntime")
}
