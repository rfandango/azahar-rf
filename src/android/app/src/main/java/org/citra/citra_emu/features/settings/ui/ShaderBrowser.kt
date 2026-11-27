// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.features.settings.ui

import android.content.Context
import android.widget.ArrayAdapter
import android.widget.ListView
import androidx.appcompat.app.AlertDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.citra.citra_emu.R

/**
 * The post-processing shader picker: the user's own shaders (.glsl files in the shaders folder)
 * at the top, and beside them a RetroArch folder with libretro's shader pack, gone through like
 * folders (a folder opens on a tap, ".." goes back up; only folders with a preset in them or
 * further down are shown). Where the pack can be used but is not there, an entry downloads it.
 *
 * [shaders] are the user's shaders by name, [presets] the pack's presets as paths in it (null
 * where presets cannot be used or the pack is not there), [offerDownload] whether to offer the
 * pack's download. [onPicked] gets the shader and the preset chosen, at most one of them not
 * empty (both empty for none); [onDownload] is called for the download, the browser closed.
 * [startFolder] is a folder of the pack to open in, null for the current preset's.
 */
class ShaderBrowser(
    private val context: Context,
    private val shaders: List<String>,
    private val presets: List<String>?,
    private val offerDownload: Boolean,
    private val currentShader: String,
    private val currentPreset: String,
    startFolder: String?,
    private val onPicked: (shader: String, preset: String) -> Unit,
    private val onDownload: () -> Unit
) {
    private sealed class Entry(val label: String) {
        object Up : Entry("..")
        class Folder(val path: String, label: String) : Entry(label)
        class Shader(val name: String, label: String) : Entry(label)
        class Preset(val path: String, label: String) : Entry(label)
        class Download(label: String) : Entry(label)
    }

    private val packName = context.getString(R.string.shader_folder_retroarch)

    /** The folder shown: null for the top, else a path in the pack ("" for its top) */
    private var folder: String? = when {
        presets == null -> null
        startFolder != null -> startFolder
        else -> currentPreset.takeIf { it.isNotEmpty() && it in presets }
            ?.substringBeforeLast('/', "")
    }

    private var entries: List<Entry> = emptyList()
    private lateinit var adapter: ArrayAdapter<String>
    private lateinit var dialog: AlertDialog

    fun show() {
        adapter = ArrayAdapter(context, android.R.layout.simple_list_item_1, mutableListOf())
        val list = ListView(context).apply { adapter = this@ShaderBrowser.adapter }
        list.setOnItemClickListener { _, _, position, _ ->
            when (val entry = entries[position]) {
                is Entry.Up -> open(folder?.let { if (it.isEmpty()) null else it.parent() })
                is Entry.Folder -> open(entry.path)
                is Entry.Shader -> {
                    dialog.dismiss()
                    onPicked(entry.name, "")
                }

                is Entry.Preset -> {
                    dialog.dismiss()
                    onPicked("", entry.path)
                }

                is Entry.Download -> {
                    dialog.dismiss()
                    onDownload()
                }
            }
        }
        dialog = MaterialAlertDialogBuilder(context)
            .setTitle(R.string.post_processing_shader)
            .setView(list)
            .setNeutralButton(R.string.post_processing_shader_none) { _, _ -> onPicked("", "") }
            .setNegativeButton(android.R.string.cancel, null)
            .create()
        open(folder)
        dialog.show()
    }

    private fun String.parent(): String = substringBeforeLast('/', "")

    private fun mark(label: String, current: Boolean) = if (current) "$label  ✓" else label

    private fun open(path: String?) {
        folder = path
        entries = if (path == null) top() else inPack(path)
        adapter.clear()
        adapter.addAll(entries.map { it.label })
        dialog.setTitle(
            when {
                path == null -> context.getString(R.string.post_processing_shader)
                path.isEmpty() -> packName
                else -> "$packName/$path"
            }
        )
    }

    private fun top(): List<Entry> = buildList {
        if (presets != null) {
            val inUse = currentPreset.isNotEmpty() && currentPreset in presets
            add(Entry.Folder("", mark("$packName/", inUse)))
        } else if (offerDownload) {
            add(Entry.Download(context.getString(R.string.shader_pack_download_item)))
        }
        shaders.sortedBy { it.lowercase() }.forEach {
            add(Entry.Shader(it, mark(it, it == currentShader)))
        }
    }

    private fun inPack(path: String): List<Entry> {
        val prefix = if (path.isEmpty()) "" else "$path/"
        val folders = sortedMapOf<String, Boolean>(String.CASE_INSENSITIVE_ORDER)
        val here = mutableListOf<Entry.Preset>()
        for (preset in presets.orEmpty()) {
            if (!preset.startsWith(prefix)) {
                continue
            }
            val rest = preset.removePrefix(prefix)
            if (rest.contains('/')) {
                // (Marked where the current preset is further down)
                val name = rest.substringBefore('/')
                folders[name] = (folders[name] ?: false) || preset == currentPreset
            } else {
                val name = rest.removeSuffix(".slangp")
                here += Entry.Preset(preset, mark(name, preset == currentPreset))
            }
        }
        return buildList {
            add(Entry.Up)
            if (path.isEmpty()) {
                add(Entry.Download(context.getString(R.string.shader_pack_update_item)))
            }
            folders.forEach { (name, inUse) -> add(Entry.Folder(prefix + name, mark("$name/", inUse))) }
            addAll(here.sortedBy { it.label.lowercase() })
        }
    }
}
