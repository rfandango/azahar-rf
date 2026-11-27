// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.utils

import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream
import org.citra.citra_emu.CitraApplication

/**
 * libretro's pack of RetroArch shaders (the "slang" ones, which the Vulkan renderer runs with
 * librashader): downloaded on request into the app's own files folder, which the native side reads
 * with plain file access (the user folder is reached through the storage framework).
 */
object ShaderPack {
    const val URL = "https://buildbot.libretro.com/assets/frontend/shaders_slang.zip"

    /** Where the pack is: its folders (crt, handheld, ...) are in it */
    val folder: File
        get() {
            val context = CitraApplication.appContext
            return File(context.getExternalFilesDir(null) ?: context.filesDir, "shaders/slang")
        }

    val isDownloaded: Boolean
        get() = folder.isDirectory && !folder.list().isNullOrEmpty()

    private const val PRESET_EXTENSION = ".slangp"
    private const val INDEX = "presets.txt"

    /**
     * The presets in the pack, as paths in it ("handheld/lcd-grid-v2.slangp"), sorted. From the
     * index written when the pack was unpacked; a pack without one is gone through once for it.
     */
    fun presets(): List<String> {
        val index = File(folder, INDEX)
        if (index.isFile) {
            return index.readLines().filter { it.isNotBlank() }
        }
        val root = folder
        val found = root.walkTopDown()
            .filter { it.isFile && it.name.endsWith(PRESET_EXTENSION, ignoreCase = true) }
            .map { it.relativeTo(root).invariantSeparatorsPath }
            .sortedBy { it.lowercase() }
            .toList()
        try {
            index.writeText(found.joinToString("\n"))
        } catch (_: IOException) {
        }
        return found
    }

    class Cancelled : IOException("cancelled")

    /**
     * Downloads the pack and puts it in place of the one there was, which stays as it was if this
     * fails or is cancelled. To be called on a background thread. [progress] gets the bytes so far
     * and the total (-1 where not known) while downloading, and (-1, -1) while unpacking.
     */
    fun download(progress: (Long, Long) -> Unit, cancelled: () -> Boolean) {
        val context = CitraApplication.appContext
        val zip = File(context.cacheDir, "shaders_slang.zip")
        try {
            val connection = URL(URL).openConnection() as HttpURLConnection
            connection.connectTimeout = 15000
            connection.readTimeout = 30000
            try {
                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw IOException("the server answered ${connection.responseCode}")
                }
                val total = connection.contentLengthLong
                connection.inputStream.use { input ->
                    FileOutputStream(zip).use { output ->
                        val buffer = ByteArray(64 * 1024)
                        var done = 0L
                        while (true) {
                            if (cancelled()) {
                                throw Cancelled()
                            }
                            val read = input.read(buffer)
                            if (read < 0) {
                                break
                            }
                            output.write(buffer, 0, read)
                            done += read
                            progress(done, total)
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }

            // Unpacked beside the pack there is, then put in its place
            progress(-1, -1)
            val target = folder
            val staging = File(target.parentFile, "slang.new")
            staging.deleteRecursively()
            staging.mkdirs()
            val stagingPath = staging.canonicalPath + File.separator
            val presets = mutableListOf<String>()
            try {
                ZipInputStream(BufferedInputStream(FileInputStream(zip))).use { entries ->
                    var entry = entries.nextEntry
                    while (entry != null) {
                        if (cancelled()) {
                            throw Cancelled()
                        }
                        val file = File(staging, entry.name).canonicalFile
                        // (Nothing written outside the folder, whatever the names in the zip)
                        if (!file.path.startsWith(stagingPath)) {
                            throw IOException(
                                "the pack has a file outside its folder: ${entry.name}"
                            )
                        }
                        if (entry.isDirectory) {
                            file.mkdirs()
                        } else {
                            file.parentFile?.mkdirs()
                            FileOutputStream(file).use { entries.copyTo(it) }
                            if (file.name.endsWith(PRESET_EXTENSION, ignoreCase = true)) {
                                presets += file.relativeTo(staging).invariantSeparatorsPath
                            }
                        }
                        entry = entries.nextEntry
                    }
                }
                File(staging, INDEX).writeText(
                    presets.sortedBy { it.lowercase() }.joinToString("\n")
                )
            } catch (e: Exception) {
                staging.deleteRecursively()
                throw e
            }
            target.deleteRecursively()
            if (!staging.renameTo(target)) {
                throw IOException("the pack could not be put in place")
            }
        } finally {
            zip.delete()
        }
    }
}
