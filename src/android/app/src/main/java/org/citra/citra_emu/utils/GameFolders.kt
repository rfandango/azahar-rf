// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.utils

import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import androidx.core.net.toUri
import androidx.preference.PreferenceManager
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.citra.citra_emu.CitraApplication

/**
 * The folders whose applications are listed, in the order they were added.
 */
object GameFolders {
    private const val KEY_GAME_PATHS = "game_paths"

    // The one folder of versions that knew no more than one. It holds the first folder of the list
    private const val KEY_GAME_PATH = "game_path"

    private val preferences
        get() = PreferenceManager.getDefaultSharedPreferences(CitraApplication.appContext)

    val folders: List<Uri>
        get() {
            val paths = try {
                Json.decodeFromString<List<String>>(
                    preferences.getString(KEY_GAME_PATHS, null) ?: "[]"
                )
            } catch (e: Exception) {
                Log.error("[GameFolders] The list of folders cannot be read: ${e.message}")
                emptyList()
            }
            // What was chosen in such a version, before this one or after it
            val single = preferences.getString(KEY_GAME_PATH, "").orEmpty()
            return (paths + single).filter { it.isNotEmpty() }.distinct().map { it.toUri() }
        }

    /** The folders in one line, for the settings */
    val summary: String
        get() = folders.joinToString(", ") { it.path.orEmpty() }

    /**
     * @return false if the folder is in the list already
     */
    fun add(folder: Uri): Boolean {
        CitraApplication.appContext.contentResolver.takePersistableUriPermission(
            folder,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        val current = folders
        if (folder in current) {
            return false
        }
        store(current + folder)
        return true
    }

    /**
     * Takes the folder off the list. Nothing in it is touched.
     */
    fun remove(folder: Uri) {
        store(folders - folder)
        // The user folder can be the very same one, and needs its permission
        if (folder == PermissionsHandler.citraDirectory) {
            return
        }
        try {
            CitraApplication.appContext.contentResolver.releasePersistableUriPermission(
                folder,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (e: SecurityException) {
            Log.warning("[GameFolders] No permission to give back for a folder: ${e.message}")
        }
    }

    /**
     * What a file or folder is, whichever of the folders it was found through: one folder of the
     * list may lie within another.
     */
    fun identify(document: Uri): String = try {
        "${document.authority}/${DocumentsContract.getDocumentId(document)}"
    } catch (_: IllegalArgumentException) {
        document.toString()
    }

    private fun store(list: List<Uri>) {
        preferences.edit()
            .putString(KEY_GAME_PATHS, Json.encodeToString(list.map { it.toString() }))
            .putString(KEY_GAME_PATH, list.firstOrNull()?.toString().orEmpty())
            .apply()
    }
}
