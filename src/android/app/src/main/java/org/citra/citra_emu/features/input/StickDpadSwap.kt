// Copyright 2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

package org.citra.citra_emu.features.input

import org.citra.citra_emu.NativeLibrary

/**
 * Where what is mapped to the circle pad and to the D-pad of the 3DS goes: to those, or, swapped, to
 * the other of the two. For devices whose left stick is in an uncomfortable place.
 *
 * Only what a controller does comes through here, not the on-screen controls.
 */
class StickDpadSwap {
    var swapped = false
        private set

    /** What the controller has, whichever way it goes. */
    private val dpadHeld = HashSet<Int>()
    private var stickX = 0f
    private var stickY = 0f

    /** The buttons of the 3DS's D-pad the stick holds down. */
    private val pressedByStick = HashSet<Int>()

    fun onButton(descriptor: String, button: Int, state: Int): Boolean {
        if (button !in DPAD) return NativeLibrary.onGamePadEvent(descriptor, button, state)

        if (state == NativeLibrary.ButtonState.PRESSED) {
            dpadHeld.add(button)
        } else {
            dpadHeld.remove(button)
        }
        return if (swapped) {
            moveCirclePadByDpad(descriptor)
        } else {
            NativeLibrary.onGamePadEvent(descriptor, button, state)
        }
    }

    fun onStick(descriptor: String, stick: Int, x: Float, y: Float): Boolean {
        if (stick != NativeLibrary.ButtonType.STICK_LEFT) {
            return NativeLibrary.onGamePadMoveEvent(descriptor, stick, x, y)
        }

        stickX = x
        stickY = y
        if (!swapped) return NativeLibrary.onGamePadMoveEvent(descriptor, stick, x, y)
        pressDpadByStick(descriptor)
        return true
    }

    /** Goes the other way from now on. What is held is let go of there and held here instead. */
    fun setSwapped(descriptor: String, value: Boolean) {
        if (value == swapped) return

        if (swapped) {
            releaseDpadOfStick(descriptor)
        } else {
            dpadHeld.forEach {
                NativeLibrary.onGamePadEvent(descriptor, it, NativeLibrary.ButtonState.RELEASED)
            }
        }
        swapped = value
        if (swapped) {
            moveCirclePadByDpad(descriptor)
            pressDpadByStick(descriptor)
        } else {
            NativeLibrary.onGamePadMoveEvent(
                descriptor,
                NativeLibrary.ButtonType.STICK_LEFT,
                stickX,
                stickY
            )
            dpadHeld.forEach {
                NativeLibrary.onGamePadEvent(descriptor, it, NativeLibrary.ButtonState.PRESSED)
            }
        }
    }

    /** The controller is not listened to any more: nothing is held from here on. */
    fun reset() {
        stickX = 0f
        stickY = 0f
        dpadHeld.clear()
        if (swapped) {
            releaseDpadOfStick("")
            moveCirclePadByDpad("")
        }
    }

    private fun moveCirclePadByDpad(descriptor: String): Boolean {
        fun held(button: Int) = if (button in dpadHeld) 1f else 0f
        // (Brought into the circle by the native side, so a diagonal is not faster.)
        return NativeLibrary.onGamePadMoveEvent(
            descriptor,
            NativeLibrary.ButtonType.STICK_LEFT,
            held(NativeLibrary.ButtonType.DPAD_RIGHT) - held(NativeLibrary.ButtonType.DPAD_LEFT),
            held(NativeLibrary.ButtonType.DPAD_DOWN) - held(NativeLibrary.ButtonType.DPAD_UP)
        )
    }

    private fun pressDpadByStick(descriptor: String) {
        for (button in DPAD) {
            val tilt = when (button) {
                NativeLibrary.ButtonType.DPAD_UP -> -stickY
                NativeLibrary.ButtonType.DPAD_DOWN -> stickY
                NativeLibrary.ButtonType.DPAD_LEFT -> -stickX
                else -> stickX
            }
            // Let go of later than pressed, or a stick held at the edge would press over and over.
            val pressed = tilt > if (button in pressedByStick) RELEASE_TILT else PRESS_TILT
            if (pressed && pressedByStick.add(button)) {
                NativeLibrary.onGamePadEvent(descriptor, button, NativeLibrary.ButtonState.PRESSED)
            } else if (!pressed && pressedByStick.remove(button)) {
                NativeLibrary.onGamePadEvent(descriptor, button, NativeLibrary.ButtonState.RELEASED)
            }
        }
    }

    private fun releaseDpadOfStick(descriptor: String) {
        pressedByStick.forEach {
            NativeLibrary.onGamePadEvent(descriptor, it, NativeLibrary.ButtonState.RELEASED)
        }
        pressedByStick.clear()
    }

    private companion object {
        /** How far the stick has to be tilted to press a direction, and to go on holding it. */
        const val PRESS_TILT = 0.5f
        const val RELEASE_TILT = 0.4f

        val DPAD = listOf(
            NativeLibrary.ButtonType.DPAD_UP,
            NativeLibrary.ButtonType.DPAD_DOWN,
            NativeLibrary.ButtonType.DPAD_LEFT,
            NativeLibrary.ButtonType.DPAD_RIGHT
        )
    }
}
