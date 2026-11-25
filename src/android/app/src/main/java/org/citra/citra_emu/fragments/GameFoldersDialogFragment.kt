// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.fragments

import android.app.Dialog
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.view.LayoutInflater
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.view.isVisible
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.citra.citra_emu.CitraApplication
import org.citra.citra_emu.R
import org.citra.citra_emu.databinding.DialogGameFoldersBinding
import org.citra.citra_emu.databinding.ListItemGameFolderBinding
import org.citra.citra_emu.utils.GameFolders
import org.citra.citra_emu.viewmodel.HomeViewModel

/**
 * The folders whose applications are listed: one more, or one less.
 */
class GameFoldersDialogFragment : DialogFragment() {
    private lateinit var binding: DialogGameFoldersBinding

    private val homeViewModel: HomeViewModel by activityViewModels()

    private val pickFolder =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { result ->
            if (result == null) {
                return@registerForActivityResult
            }
            val added = GameFolders.add(result)
            Toast.makeText(
                CitraApplication.appContext,
                if (added) R.string.games_dir_selected else R.string.games_folder_already_added,
                Toast.LENGTH_LONG
            ).show()
            if (added) {
                onFoldersChanged()
            }
        }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        binding = DialogGameFoldersBinding.inflate(LayoutInflater.from(requireContext()))
        showFolders()
        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.select_games_folder)
            .setMessage(R.string.games_folders_message)
            .setView(binding.root)
            .setPositiveButton(R.string.add_games_folder, null)
            .setNegativeButton(R.string.close, null)
            .create()
    }

    override fun onStart() {
        super.onStart()
        // Set here, as the dialog would close at the press of a button set when it is made
        (requireDialog() as AlertDialog).getButton(AlertDialog.BUTTON_POSITIVE)
            .setOnClickListener { pickFolder.launch(null) }
    }

    private fun showFolders() {
        val folders = GameFolders.folders
        binding.folders.removeAllViews()
        binding.textEmpty.isVisible = folders.isEmpty()
        folders.forEach { folder ->
            val row = ListItemGameFolderBinding.inflate(
                LayoutInflater.from(requireContext()),
                binding.folders,
                true
            )
            val path = pathOf(folder)
            val name = path.substringAfterLast('/').substringAfterLast(':')
            row.textName.text = name.ifEmpty { path }
            row.textPath.text = path
            row.buttonRemove.setOnClickListener { askToRemove(folder, path) }
        }
    }

    // The folder picker's button to allow a folder lies about where a folder's button here is: a
    // second press on it would take the folder that has just been added off the list again
    private fun askToRemove(folder: Uri, path: String) {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.remove_games_folder)
            .setMessage(path)
            .setPositiveButton(R.string.remove_games_folder) { _, _ ->
                GameFolders.remove(folder)
                onFoldersChanged()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun onFoldersChanged() {
        showFolders()
        homeViewModel.onGameFoldersChanged(requireActivity())
    }

    /** Such as "primary:Games/3DS" */
    private fun pathOf(folder: Uri): String = try {
        DocumentsContract.getTreeDocumentId(folder)
    } catch (_: IllegalArgumentException) {
        folder.path.orEmpty()
    }

    companion object {
        const val TAG = "GameFoldersDialogFragment"
    }
}
