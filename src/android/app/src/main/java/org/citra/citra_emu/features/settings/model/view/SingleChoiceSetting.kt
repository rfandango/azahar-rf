// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.features.settings.model.view

import android.content.res.Resources
import androidx.annotation.StringRes
import org.citra.citra_emu.R
import org.citra.citra_emu.features.settings.model.AbstractSetting
import org.citra.citra_emu.features.settings.model.CitraSettings

class SingleChoiceSetting(
    val settings: CitraSettings,
    setting: AbstractSetting<*>?,
    titleId: Int,
    descriptionId: Int,
    val choicesId: Int,
    val valuesId: Int,
    val key: String? = null,
    val defaultValue: Int? = null,
    override var isEnabled: Boolean = true,
    private val getValue: (() -> Int)? = null,
    private val setValue: ((Int) -> Unit)? = null,
    @StringRes override var disabledMessage: Int =
        R.string.setting_disabled_description_incompatible_setting,
    // Choices that are put together, instead of those of choicesId and valuesId
    private val ownChoices: Array<String>? = null,
    private val ownValues: IntArray? = null
) : SettingsItem(setting, titleId, descriptionId) {
    override val type = TYPE_SINGLE_CHOICE

    fun choices(resources: Resources): Array<String> =
        ownChoices ?: resources.getStringArray(choicesId)

    /**
     * The value of each choice, or null if that is its place in the list.
     */
    fun values(resources: Resources): IntArray? =
        ownValues ?: if (valuesId > 0) resources.getIntArray(valuesId) else null

    val selectedValue: Int
        get() {
            if (getValue != null) {
                return getValue.invoke()
            }
            @Suppress("UNCHECKED_CAST")
            val s = (setting as? AbstractSetting<Int>) ?: return defaultValue!!
            return settings.get(s)
        }

    /**
     * Write a value to the backing int .
     * @param selection New value of the int.
     */
    fun setSelectedValue(selection: Int) {
        if (setValue != null) {
            setValue(selection)
        } else {
            @Suppress("UNCHECKED_CAST")
            val backSetting = setting as AbstractSetting<Int>
            settings.setBySettingsContext(backSetting, selection)
        }
    }
}
