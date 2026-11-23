// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Puts a GPU driver into the app, which then uses it by default on Adreno GPUs. The driver is not
// part of the source: it is taken from a driver package (the zip the driver manager installs), see
// gpu_driver/README.md. Without one the app comes without a driver.

import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.util.zip.ZipFile

val driverFolder = file("gpu_driver")

/** The package to take: what the property gpuDriver names, or the only one in the folder. */
fun findDriverPackage(): File? {
    val property = providers.gradleProperty("gpuDriver").orNull?.trim()
    if (property == null) {
        val packages = driverFolder.listFiles { f: File -> f.isFile && f.extension == "zip" }
            .orEmpty().sortedBy { it.name }
        if (packages.size > 1) {
            throw GradleException(
                "More than one GPU driver package in $driverFolder " +
                    "(${packages.joinToString { it.name }}): leave one, or choose with -PgpuDriver"
            )
        }
        return packages.firstOrNull()
    }
    if (property.isEmpty() || property == "none") {
        return null
    }
    val path = if (property.startsWith("~/")) {
        System.getProperty("user.home") + property.drop(1)
    } else {
        property
    }
    return gradle.startParameter.currentDir.resolve(path).also {
        if (!it.isFile) {
            throw GradleException("gpuDriver: there is no file $it")
        }
    }
}

val driverPackage = findDriverPackage()

tasks.register("injectGpuDriver") {
    val output = layout.buildDirectory.dir("generated/gpuDriver").get().asFile
    inputs.property("package", driverPackage?.path ?: "none")
    driverPackage?.let { inputs.file(it) }
    outputs.dir(output)

    doLast {
        output.deleteRecursively()
        val libraries = output.resolve("jniLibs/arm64-v8a").apply { mkdirs() }
        val assets = output.resolve("assets/gpu_driver").apply { mkdirs() }
        val source = driverPackage
        if (source == null) {
            logger.lifecycle("GPU driver in the app: none")
            return@doLast
        }

        ZipFile(source).use { zip ->
            fun fail(problem: String): Nothing =
                throw GradleException("GPU driver package $source: $problem")

            fun read(name: String): ByteArray {
                val entry = zip.getEntry(name) ?: fail("no $name in it")
                return zip.getInputStream(entry).use { it.readBytes() }
            }

            val metadata = (JsonSlurper().parse(read("meta.json")) as Map<*, *>).toMutableMap()
            val name = metadata["name"] as? String ?: fail("its meta.json has no name")
            val library = metadata["libraryName"] as? String
                ?: fail("its meta.json has no libraryName")
            val code = read(library)
            // ELF, 64 bits, AArch64
            if (code.size < 20 || code[0] != 0x7F.toByte() || code[1] != 'E'.code.toByte() ||
                code[4] != 2.toByte() || code[18] != 0xB7.toByte() || code[19] != 0.toByte()
            ) {
                fail("$library is not an arm64 library")
            }
            // Android only installs libraries that are named like one
            val installedName = if (library.startsWith("lib")) library else "lib$library"
            if (!installedName.endsWith(".so") || installedName.contains('/')) {
                fail("$library cannot be installed under that name")
            }
            metadata["libraryName"] = installedName
            libraries.resolve(installedName).writeBytes(code)
            assets.resolve("meta.json")
                .writeText(JsonOutput.prettyPrint(JsonOutput.toJson(metadata)) + "\n")
            logger.lifecycle("GPU driver in the app: $name, from $source")
        }
    }
}

tasks.named("preBuild") {
    dependsOn("injectGpuDriver")
}
