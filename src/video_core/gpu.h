// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <boost/serialization/access.hpp>

#include "core/hle/result.h"
#include "core/hle/service/gsp/gsp_interrupt.h"

namespace Service::GSP {
struct Command;
struct FrameBufferInfo;
} // namespace Service::GSP

namespace Core {
class System;
}

namespace Pica {
class DebugContext;
class PicaCore;
struct RegsLcd;
union ColorFill;
} // namespace Pica

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

/// Measured on hardware to be 2240568 timer cycles or 4481136 ARM11 cycles
constexpr u64 FRAME_TICKS = 4481136ull;

class GraphicsDebugger;
class FrameLiveness;
class RendererBase;
class RightEyeDisabler;
class WorkRules;

/**
 * The GPU class is the high level interface to the video_core for core services.
 */
class GPU {
public:
    explicit GPU(Core::System& system, Frontend::EmuWindow& emu_window,
                 Frontend::EmuWindow* secondary_window);
    ~GPU();

    /// Sets the function to call for signalling GSP interrupts.
    void SetInterruptHandler(Service::GSP::InterruptHandler handler);

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(PAddr addr, u32 size);

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(PAddr addr, u32 size);

    /// Flushes and invalidates all memory in the rasterizer cache and removes any leftover state.
    void ClearAll(bool flush);

    /// Executes the provided GSP command.
    Result Execute(const Service::GSP::Command& command);

    /**
     * Has the command executed: on the thread that decodes commands if there is one (lists,
     * fills and transfers; the game goes on meanwhile and the result is success), at once
     * otherwise. The interrupt that tells the game a command is done is raised on the
     * emulation thread, when the command is done: see DeliverInterrupts. Emulation thread.
     */
    Result Submit(const Service::GSP::Command& command);

    /// What the emulation thread waits for the decoding thread for, see WaitIdle
    enum class WaitReason : u32 {
        Other,
        Command,      ///< A command that is not handed over is to be executed
        Register,     ///< A register of the emulated GPU is read or written
        BufferSwap,   ///< The game says which of its buffers is to be shown
        VBlank,       ///< A frame is shown
        MemoryAccess, ///< The game reads or writes memory the rasterizer keeps
        MapPages,     ///< The game's memory is mapped anew
        NothingToRun, ///< No thread of the game has anything to do
        // A memory access, by what the game touches (Memory::MemorySystem::Impl::AccessKind)
        ReadGpuNewer,         ///< A read of what a surface holds newer than memory
        ReadGpuWritten,       ///< A read of a page that what is still to be decoded may write
        ReadWatched,          ///< A read of a page a draw reads in place
        WriteSurface,         ///< A write into what a surface covers
        WriteWatched,         ///< A write to a page a draw reads in place
        WriteVertexData,      ///< A write to vertex data the decoder read before
        WriteIndexData,       ///< To index data
        WriteTexture,         ///< To a texture it loaded
        WriteCalledList,      ///< To a command list called from others
        WriteOtherReference,  ///< To something else it read
        WriteBesideReference, ///< To a page it reads from, but not to what it read of it
        WriteGpuWritten,      ///< To a page that what is still to be decoded may write
        Count,
    };

    /// Returns when nothing that was handed to Submit is left to do. Everything the
    /// emulation thread does with the emulated GPU's state goes through this first, so that
    /// only one thread at a time is at it. Does nothing on the decoding thread itself.
    void WaitIdle(WaitReason reason = WaitReason::Other);

    /// Raises the interrupts of the commands that are done since this was last called.
    /// Emulation thread only.
    void DeliverInterrupts();

    /// Whether commands handed to Submit are still on their way
    [[nodiscard]] bool HasPendingWork() const;

    /// Counts an access for the decoder's line in the log that did not have to wait
    void NoteAccess(WaitReason reason);

    /// Whether the game is told a command is done when it hands it over (design B): then no
    /// thread of the game waits for the decoder, and emulated time may run on while it works
    [[nodiscard]] bool CompletesEarly() const;

    /// Updates GPU display framebuffer configuration using the specified parameters.
    void SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info);

    /// Sets the LCD color fill configuration for the top and bottom screens.
    void SetColorFill(const Pica::ColorFill& fill);

    /// Reads a word from the GPU virtual address.
    u32 ReadReg(VAddr addr);

    /// Writes the provided value to the GPU virtual address.
    void WriteReg(VAddr addr, u32 data);

    /// Returns a mutable reference to the renderer.
    [[nodiscard]] VideoCore::RendererBase& Renderer();

    /// Returns a mutable reference to the PICA GPU.
    [[nodiscard]] Pica::PicaCore& PicaCore();

    /// Returns an immutable reference to the PICA GPU.
    [[nodiscard]] const Pica::PicaCore& PicaCore() const;

    /// Returns a mutable reference to the GSP command debugger.
    [[nodiscard]] GraphicsDebugger& Debugger();

    RightEyeDisabler& GetRightEyeDisabler() {
        return *right_eye_disabler;
    }

    FrameLiveness& GetFrameLiveness() {
        return *frame_liveness;
    }

    void ApplyPerProgramSettings(u64 program_ID);

    /// Reports a transfer as done whose completion is being held for the VBlank (see
    /// MemoryTransfer). The hold is no part of a save state, so this is called before one is
    /// made: the game in it would wait for good otherwise.
    void FinishHeldTransfer();

#ifdef ENABLE_FREECAM
    /// Sets the free camera transform, a transform matrix applied to the position
    /// of every vertex.
    /// skip_2d leaves vertices with w == ~1 (most 2D elements) untouched.
    void SetFreeCam(const std::array<f32, 16>& matrix, bool skip_2d);
#endif

    /// Recreates the renderer (for GL context reset in libretro)
    void RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window);

    /// Releases the renderer (for GL context destroy in libretro)
    void ReleaseRenderer();

private:
    void DecoderLoop();
    void StopDecoder();
    void ReportDecoderWaits();
    /// Whether commands go to the decoder's thread now: not while a movie is recorded or
    /// played back
    [[nodiscard]] bool UsesDecoder() const;
    /// Waits for the image of the VBlank before to have been composed
    void WaitForPresent();
    /// The buffer of the top screen a transfer or copy fills, or 0: for the pacing rule
    PAddr PaceTargetOf(const Service::GSP::Command& command);

    void SubmitCmdList(u32 index);

    // Interrupt index must be 0 or 1 to signal the relative PSC interrupt.
    void MemoryFill(u32 index, u32 intr_index);

    void MemoryTransfer();

    void VBlankCallback(uintptr_t user_data, s64 cycles_late);

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version);

    std::unique_ptr<RightEyeDisabler> right_eye_disabler;
    /// The rules that find work that need not be done; the one of them that follows results
    /// to the screen, which is also told of what is displayed and of what the game reads
    std::unique_ptr<WorkRules> work_rules;
    FrameLiveness* frame_liveness{};

    /// Pacing of a game whose render loop does not wait for the display, see MemoryTransfer.
    /// Not part of a save state: it settles again within a few frames, and what is held is let
    /// go before a state is made.
    struct RenderPace {
        /// Buffers the game has shown the top screen from lately (left eye)
        std::array<PAddr, 4> shown{};
        u32 shown_next{};
        /// The one of them an image was copied into since the last VBlank, or 0
        PAddr filled{};
        /// The one whose copy is being held for the VBlank, or 0
        PAddr held{};
        u64 held_delay{};
        bool reported{};
        /// Bytes a row of the top screen's buffers, where the game is told "done" at
        /// hand-over: then all of this is the emulation thread's, see GPU::Submit
        u32 stride{};
    };
    RenderPace render_pace;

private:
    friend class FrameLiveness;
    friend class RightEyeDisabler;
    struct Impl;
    std::unique_ptr<Impl> impl;

    /// Translates the address of a GPU buffer as the GSP module does, or returns std::nullopt when
    /// the buffer is not in GPU visible memory.
    std::optional<PAddr> VirtualToPhysicalAddress(VAddr addr, u32 size);
};

} // namespace VideoCore
