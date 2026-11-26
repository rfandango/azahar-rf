// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cstring>
#include <optional>
#include <span>
#include <dynarmic/interface/A32/a32.h>
#include <dynarmic/interface/code_cache.h>
#include <dynarmic/interface/optimization_flags.h>
#include <fmt/format.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/trace_markers.h"
#include "common/arch.h"
#if CITRA_ARCH(arm64)
#include "core/arm/dynarmic/ahead_translation.h"
#endif
#include "core/arm/dynarmic/arm_dynarmic.h"
#include "core/arm/dynarmic/arm_dynarmic_cp15.h"
#include "core/arm/dynarmic/arm_exclusive_monitor.h"
#include "core/arm/dynarmic/arm_tick_counts.h"
#include "core/arm/dynarmic/jit_code_cache.h"
#include "core/arm/exception_handler.h"
#include "core/core.h"
#include "core/core_timing.h"
#ifdef ENABLE_GDBSTUB
#include "core/gdbstub/gdbstub.h"
#endif
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/process.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/vm_manager.h"
#include "core/memory.h"

#ifndef SIGILL
constexpr u32 SIGILL = 4;
#endif

#ifndef SIGTRAP
constexpr u32 SIGTRAP = 5;
#endif

namespace Core {

class DynarmicUserCallbacks final : public Dynarmic::A32::UserCallbacks {
public:
    explicit DynarmicUserCallbacks(ARM_Dynarmic& parent)
        : parent(parent), svc_context(parent.system), memory(parent.memory) {}
    ~DynarmicUserCallbacks() = default;

    std::optional<std::uint32_t> MemoryReadCode(VAddr vaddr) override {
        return memory.Read32OrNullopt(vaddr);
    }

    bool MemoryHoldsCode(VAddr vaddr, const std::uint32_t* words, std::size_t count) override {
        auto& pointers = parent.current_page_table->GetPointerArray();
        while (count != 0) {
            const u32 offset = vaddr & Memory::CITRA_PAGE_MASK;
            const std::size_t here =
                std::min<std::size_t>(count, (Memory::CITRA_PAGE_SIZE - offset) / 4);
            if (const u8* const page = pointers[vaddr >> Memory::CITRA_PAGE_BITS]) {
                if (std::memcmp(page + offset, words, here * 4) != 0) {
                    return false;
                }
            } else if (!UserCallbacks::MemoryHoldsCode(vaddr, words, here)) {
                return false;
            }
            vaddr += static_cast<u32>(here * 4);
            words += here;
            count -= here;
        }
        return true;
    }

    std::uint8_t MemoryRead8(VAddr vaddr) override {
        return memory.Read8(vaddr);
    }
    std::uint16_t MemoryRead16(VAddr vaddr) override {
        return memory.Read16(vaddr);
    }
    std::uint32_t MemoryRead32(VAddr vaddr) override {
        return memory.Read32(vaddr);
    }
    std::uint64_t MemoryRead64(VAddr vaddr) override {
        return memory.Read64(vaddr);
    }

    void MemoryWrite8(VAddr vaddr, std::uint8_t value) override {
        memory.Write8(vaddr, value);
    }
    void MemoryWrite16(VAddr vaddr, std::uint16_t value) override {
        memory.Write16(vaddr, value);
    }
    void MemoryWrite32(VAddr vaddr, std::uint32_t value) override {
        memory.Write32(vaddr, value);
    }
    void MemoryWrite64(VAddr vaddr, std::uint64_t value) override {
        memory.Write64(vaddr, value);
    }

    /// The word at `vaddr` is of the running process's code or of the constants loaded with
    /// it, in memory the process cannot write. Nothing else: memory the system writes and the
    /// process only reads (the shared page with the clock, the configuration block, a block
    /// shared with a service) is read-only to the game and changes all the same. Whoever
    /// patches such a word in the game's place (loading a module, a cheat) says so to the
    /// recompiler by its address (Core::System::InvalidateCacheRange).
    bool IsReadOnlyMemory(VAddr vaddr) override {
        if (!parent.fold_read_only_loads) {
            return false;
        }
        const std::shared_ptr<Kernel::Process>& process =
            parent.system.Kernel().GetCurrentProcess();
        if (!process) {
            return false;
        }
        const auto vma = process->vm_manager.FindVMA(vaddr);
        if (vma == process->vm_manager.vma_map.end()) {
            return false;
        }
        const Kernel::VirtualMemoryArea& area = vma->second;
        return area.type == Kernel::VMAType::BackingMemory &&
               area.meminfo_state == Kernel::MemoryState::Code &&
               (static_cast<u8>(area.permissions) &
                static_cast<u8>(Kernel::VMAPermission::Write)) == 0 &&
               u64{vaddr} + 4 <= u64{area.base} + area.size;
    }

    bool MemoryWriteExclusive8(u32 vaddr, u8 value, u8 expected) override {
        return memory.WriteExclusive8(vaddr, value, expected);
    }
    bool MemoryWriteExclusive16(u32 vaddr, u16 value, u16 expected) override {
        return memory.WriteExclusive16(vaddr, value, expected);
    }
    bool MemoryWriteExclusive32(u32 vaddr, u32 value, u32 expected) override {
        return memory.WriteExclusive32(vaddr, value, expected);
    }
    bool MemoryWriteExclusive64(u32 vaddr, u64 value, u64 expected) override {
        return memory.WriteExclusive64(vaddr, value, expected);
    }

    void InterpreterFallback(VAddr pc, std::size_t num_instructions) override {
        // Should never happen.
        UNREACHABLE_MSG("InterpeterFallback reached with pc = 0x{:08x}, code = 0x{:08x}, num = {}",
                        pc, MemoryReadCode(pc).value(), num_instructions);
    }

    void CallSVC(std::uint32_t swi) override {
        svc_context.CallSVC(swi);
    }

    void ExceptionRaised(VAddr pc, Dynarmic::A32::Exception exception) override {
        switch (exception) {
        case Dynarmic::A32::Exception::UndefinedInstruction:
        case Dynarmic::A32::Exception::UnpredictableInstruction:
        case Dynarmic::A32::Exception::DecodeError:
        case Dynarmic::A32::Exception::NoExecuteFault:
            break;
        case Dynarmic::A32::Exception::Breakpoint:
#ifdef ENABLE_GDBSTUB
            if (GDBStub::IsConnected()) {
                parent.SetPC(pc);
                parent.ServeBreak(SIGTRAP);
                return;
            }
#endif
            break;
        case Dynarmic::A32::Exception::SendEvent:
        case Dynarmic::A32::Exception::SendEventLocal:
        case Dynarmic::A32::Exception::WaitForInterrupt:
        case Dynarmic::A32::Exception::WaitForEvent:
        case Dynarmic::A32::Exception::Yield:
        case Dynarmic::A32::Exception::PreloadData:
        case Dynarmic::A32::Exception::PreloadDataWithIntentToWrite:
        case Dynarmic::A32::Exception::PreloadInstruction:
            return;
        }

        static constexpr auto ExceptionToString = [](Dynarmic::A32::Exception e) -> std::string {
            switch (e) {
            case Dynarmic::A32::Exception::UndefinedInstruction:
                return "UndefinedInstruction";
            case Dynarmic::A32::Exception::UnpredictableInstruction:
                return "UnpredictableInstruction";
            case Dynarmic::A32::Exception::DecodeError:
                return "DecodeError";
            case Dynarmic::A32::Exception::NoExecuteFault:
                return "NoExecuteFault";
            case Dynarmic::A32::Exception::Breakpoint:
                return "Breakpoint";
            default:
                return fmt::format("Unknown({})", e);
            }
        };

        parent.SetPC(pc);
#ifdef ENABLE_GDBSTUB
        if (GDBStub::IsConnected()) {
            parent.ServeBreak(SIGILL);
        } else
#endif
        {
            Core::ExceptionType exc_type;
            switch (exception) {
            case Dynarmic::A32::Exception::UndefinedInstruction:
            case Dynarmic::A32::Exception::UnpredictableInstruction:
            case Dynarmic::A32::Exception::DecodeError:
            case Dynarmic::A32::Exception::NoExecuteFault:
                exc_type = Core::ExceptionType::UndefinedInstruction;
                break;
            case Dynarmic::A32::Exception::Breakpoint:
                exc_type = Core::ExceptionType::Break;
                break;
            default:
                exc_type = Core::ExceptionType::UndefinedInstruction;
                break;
            }
            Core::LogException(parent.system, exc_type);
        }
    }

#if CITRA_ARCH(arm64)
    void TranslatedOnDemand(std::uint64_t location) override {
        if (AheadTranslation* const ahead = parent.system.GetAheadTranslation()) {
            ahead->NoteOnDemand(location, parent.system.Kernel().GetCurrentProcess().get());
        }
    }
#endif

    void AddTicks(std::uint64_t ticks) override {
        Common::TraceMarkers::NoteGuestTicks(ticks);
        parent.GetTimer().AddTicks(ticks);
    }
    std::uint64_t GetTicksRemaining() override {
        s64 ticks = parent.GetTimer().GetDowncount();
        return static_cast<u64>(ticks <= 0 ? 0 : ticks);
    }
    std::uint64_t GetTicksForCode(bool is_thumb, VAddr pc, std::uint32_t instruction) override {
#ifdef CITRA_ENABLE_TRACE_MARKERS
        NoteHintInstruction(is_thumb, pc, instruction);
#endif
        if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::CountGuestInstructions) !=
            0) [[unlikely]] {
            return 1;
        }
        return Core::TicksForInstruction(is_thumb, instruction);
    }

#ifdef CITRA_ENABLE_TRACE_MARKERS
    /// Says which of the instructions a program waits with (YIELD, WFE, WFI, SEV) the
    /// recompiler comes across: it translates them to nothing, so a program that waits in a
    /// loop of them runs that loop for all the time it is given.
    static void NoteHintInstruction(bool is_thumb, VAddr pc, std::uint32_t instruction) {
        const std::uint32_t hint =
            is_thumb ? ((instruction & 0xFF0F) == 0xBF00 ? (instruction >> 4) & 0xF : 0)
                     : ((instruction & 0x0FFFFFF0) == 0x0320F000 ? instruction & 0xF : 0);
        if (hint < 1 || hint > 4) {
            return;
        }
        static constexpr std::array<const char*, 5> names{"", "YIELD", "WFE", "WFI", "SEV"};
        static std::array<std::atomic<u32>, 5> counts{};
        const u32 count = ++counts[hint];
        // The first few of each, then every time the count doubles
        if (count <= 4 || (count & (count - 1)) == 0) {
            LOG_INFO(Core_ARM11,
                     "Guest code waits with {} at {:08X} ({}): {} places translated so far",
                     names[hint], pc, is_thumb ? "Thumb" : "ARM", count);
        }
    }
#endif

    ARM_Dynarmic& parent;
    Kernel::SVCContext svc_context;
    Memory::MemorySystem& memory;
};

/// The title's code cache as one core's recompilers see it. The code they produce refers to
/// that core's coprocessor state by address; here those addresses become positions in the
/// state, which mean the same for every core and in every run.
class DynarmicCodeCache final : public Dynarmic::CodeCache {
public:
    explicit DynarmicCodeCache(JitCodeCache& store_, const CP15State& cp15_state)
        : store{store_}, cp15_base{reinterpret_cast<u64>(&cp15_state)} {}

    std::span<const std::byte> Lookup(std::uint64_t location, std::size_t index) override {
        return store.Lookup(location, index);
    }

    void Store(std::uint64_t location, std::span<const std::byte> block) override {
        store.Store(location, block);
    }

    std::optional<std::uint64_t> PointerToId(std::uint64_t pointer) override {
        if (pointer < cp15_base || pointer - cp15_base >= sizeof(CP15State)) {
            return std::nullopt;
        }
        return pointer - cp15_base;
    }

    std::optional<std::uint64_t> IdToPointer(std::uint64_t id) override {
        if (id >= sizeof(CP15State)) {
            return std::nullopt;
        }
        return cp15_base + id;
    }

private:
    JitCodeCache& store;
    u64 cp15_base;
};

ARM_Dynarmic::ARM_Dynarmic(Core::System& system_, Memory::MemorySystem& memory_, u32 core_id_,
                           std::shared_ptr<Core::Timing::Timer> timer_,
                           Core::ExclusiveMonitor& exclusive_monitor_)
    : ARM_Interface(core_id_, timer_), system(system_), memory(memory_),
      cb(std::make_unique<DynarmicUserCallbacks>(*this)),
      exclusive_monitor{dynamic_cast<Core::DynarmicExclusiveMonitor&>(exclusive_monitor_)} {
    SetPageTable(memory.GetCurrentPageTable());
}

ARM_Dynarmic::~ARM_Dynarmic() = default;

MICROPROFILE_DEFINE(ARM_Jit, "ARM JIT", "ARM JIT", MP_RGB(255, 64, 64));

void ARM_Dynarmic::Run() {
    ASSERT(memory.GetCurrentPageTable() == current_page_table);
    MICROPROFILE_SCOPE(ARM_Jit);
    CITRA_TRACE_SCOPE_DETAIL("ARM::Run");
    if (break_flag) [[unlikely]] {
        return;
    }

    jit->Run();
}

void ARM_Dynarmic::Step() {
    if (break_flag) [[unlikely]] {
        return;
    }

    jit->Step();
}

void ARM_Dynarmic::SetPC(u32 pc) {
    jit->Regs()[15] = pc;
}

u32 ARM_Dynarmic::GetPC() const {
    return jit->Regs()[15];
}

u32 ARM_Dynarmic::GetReg(int index) const {
    return jit->Regs()[index];
}

void ARM_Dynarmic::SetReg(int index, u32 value) {
    jit->Regs()[index] = value;
}

u32 ARM_Dynarmic::GetVFPReg(int index) const {
    return jit->ExtRegs()[index];
}

void ARM_Dynarmic::SetVFPReg(int index, u32 value) {
    jit->ExtRegs()[index] = value;
}

u32 ARM_Dynarmic::GetVFPSystemReg(VFPSystemRegister reg) const {
    switch (reg) {
    case VFP_FPSCR:
        return jit->Fpscr();
    case VFP_FPEXC:
        return fpexc;
    default:
        UNREACHABLE_MSG("Unknown VFP system register: {}", reg);
    }

    return UINT_MAX;
}

void ARM_Dynarmic::SetVFPSystemReg(VFPSystemRegister reg, u32 value) {
    switch (reg) {
    case VFP_FPSCR:
        jit->SetFpscr(value);
        return;
    case VFP_FPEXC:
        fpexc = value;
        return;
    default:
        UNREACHABLE_MSG("Unknown VFP system register: {}", reg);
    }
}

u32 ARM_Dynarmic::GetCPSR() const {
    return jit->Cpsr();
}

void ARM_Dynarmic::SetCPSR(u32 cpsr) {
    jit->SetCpsr(cpsr);
}

u32 ARM_Dynarmic::GetCP15Register(CP15Register reg) const {
    switch (reg) {
    case CP15_THREAD_UPRW:
        return cp15_state.cp15_thread_uprw;
    case CP15_THREAD_URO:
        return cp15_state.cp15_thread_uro;
    default:
        UNREACHABLE_MSG("Unknown CP15 register: {}", reg);
    }

    return 0;
}

void ARM_Dynarmic::SetCP15Register(CP15Register reg, u32 value) {
    switch (reg) {
    case CP15_THREAD_UPRW:
        cp15_state.cp15_thread_uprw = value;
        return;
    case CP15_THREAD_URO:
        cp15_state.cp15_thread_uro = value;
        return;
    default:
        UNREACHABLE_MSG("Unknown CP15 register: {}", reg);
    }
}

void ARM_Dynarmic::SaveContext(ThreadContext& ctx) {
    ctx.cpu_registers = jit->Regs();
    ctx.cpsr = jit->Cpsr();
    ctx.fpu_registers = jit->ExtRegs();
    ctx.fpscr = jit->Fpscr();
    ctx.fpexc = fpexc;
}

void ARM_Dynarmic::LoadContext(const ThreadContext& ctx) {
    jit->Regs() = ctx.cpu_registers;
    jit->SetCpsr(ctx.cpsr);
    jit->ExtRegs() = ctx.fpu_registers;
    jit->SetFpscr(ctx.fpscr);
    fpexc = ctx.fpexc;
}

void ARM_Dynarmic::PrepareReschedule() {
    if (jit->IsExecuting()) {
        jit->HaltExecution();
    }
}

void ARM_Dynarmic::ClearInstructionCache() {
    for (const auto& j : jits) {
        j.second->ClearCache();
    }
}

void ARM_Dynarmic::InvalidateCacheRange(u32 start_address, std::size_t length) {
    jit->InvalidateCacheRange(start_address, length);
}

void ARM_Dynarmic::DumpBlockLocations(std::FILE* out) const {
    for (const auto& [page_table, table_jit] : jits) {
        for (const Dynarmic::A32::Jit::BlockLocation& block : table_jit->GetBlockLocations()) {
            fmt::print(out, "{:x} {:x} {:x} {:x}\n", GetID(), block.host_address, block.host_size,
                       block.location);
        }
    }
}

void ARM_Dynarmic::DumpRecompiledCode(std::FILE* out) const {
    const auto write = [out](const auto& value) { std::fwrite(&value, sizeof(value), 1, out); };
    for (const auto& [page_table, table_jit] : jits) {
        for (const Dynarmic::A32::Jit::BlockLocation& block : table_jit->GetBlockLocations()) {
            write(u32{0x4B4C4252}); // "RBLK"
            write(static_cast<u32>(GetID()));
            write(static_cast<u64>(block.host_address));
            write(static_cast<u64>(block.location));
            write(static_cast<u32>(block.host_size));
            write(static_cast<u32>(block.instruction_offsets.size()));
            for (const auto& [offset, pc] : block.instruction_offsets) {
                write(static_cast<u32>(offset));
                write(static_cast<u32>(pc));
            }
            std::fwrite(reinterpret_cast<const void*>(block.host_address), 1, block.host_size, out);
        }
    }
}

void ARM_Dynarmic::ClearExclusiveState() {
    jit->ClearExclusiveState();
}

std::shared_ptr<Memory::PageTable> ARM_Dynarmic::GetPageTable() const {
    return current_page_table;
}

void ARM_Dynarmic::SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) {
    // The table that is set already means the same recompiler with the registers it has:
    // nothing to save, look up and load back. This is called at every switch between the
    // emulated cores, which mostly run threads of one process.
    if (jit != nullptr && current_page_table == page_table &&
        (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyEmulationOverheads) ==
            0) {
        return;
    }
    current_page_table = page_table;
    ThreadContext ctx{};
    if (jit) {
        SaveContext(ctx);
    }

    auto iter = jits.find(current_page_table);
    if (iter != jits.end()) {
        jit = iter->second.get();
        LoadContext(ctx);
        return;
    }

    auto new_jit = MakeJit();
    jit = new_jit.get();
    LoadContext(ctx);
    jits.emplace(current_page_table, std::move(new_jit));
}

void ARM_Dynarmic::ServeBreak([[maybe_unused]] int signal) {
#ifdef ENABLE_GDBSTUB
    GDBStub::Break(signal);
#endif
}

std::unique_ptr<Dynarmic::A32::Jit> ARM_Dynarmic::MakeJit() {
    Dynarmic::A32::UserConfig config;
    config.callbacks = cb.get();
    if (!code_cache) {
        if (JitCodeCache* const store = system.GetJitCodeCache()) {
            code_cache = std::make_unique<DynarmicCodeCache>(*store, cp15_state);
        }
    }
    config.code_cache = code_cache.get();
    if ((Common::TraceMarkers::Experiments3() & Common::TraceMarkers::CountGuestInstructions) != 0)
        [[unlikely]] {
        // The stored code counts time as it was made to
        config.code_cache = nullptr;
    }
    if (current_page_table) {
        config.page_table = &current_page_table->GetPointerArray();
        // Host MMU access, if it is in use: an access goes straight to the address in the
        // host's view of the address space, and where that faults (nothing of the emulated
        // system's memory is there) the block is made again with a look into the page table
        // for that access
        if (u8* const view = current_page_table->pointers.HostView()) {
            config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(view);
            config.fastmem_guard_size = current_page_table->pointers.HostViewGuard();
            config.page_access = current_page_table->pointers.HostViewAccess();
        }
    }
    config.coprocessors[15] = std::make_shared<DynarmicCP15>(cp15_state);
    config.define_unpredictable_behaviour = true;

    // The recompiler's dispatcher looks in its own table of recent blocks before it calls out
    // for one; a profiling switch has it call out every time, as it used to
    if (Common::TraceMarkers::Experiments2() & Common::TraceMarkers::LegacyDispatcher) {
        config.optimizations &= ~Dynarmic::OptimizationFlag::FastDispatch;
    }

    // Where the guest's registers are while its code runs: in the host's, unless the
    // profiling switch asks for the state in memory as it used to be
    config.pin_guest_registers =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::GuestRegistersInMemory) == 0;
    config.write_into_guest_registers =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::GuestRegisterMoves) == 0;
    config.fold_conditional_branches =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::LoneConditionalBranches) == 0;
    config.guest_flags_in_host_flags =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::GuestFlagsInMemory) == 0;
    config.fp_status_in_host =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::FpStatusInMemory) == 0;
    config.fp_registers_through_general =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::FpRegistersThroughFp) == 0;
    config.look_up_blocks_in_place =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ReturnPredictionList) == 0;
    config.pin_guest_fp_registers = (Common::TraceMarkers::Experiments3() &
                                     Common::TraceMarkers::GuestFpRegistersInMemory) == 0;
    config.conditional_instructions_in_blocks =
        (Common::TraceMarkers::Experiments3() &
         Common::TraceMarkers::ConditionalInstructionsSplitBlocks) == 0;
    // Loads of constants next to the code are made as constants, unless the profiling switch
    // asks for them as loads
    fold_read_only_loads =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::LegacyLiteralLoads) == 0;
    config.fold_read_only_loads = fold_read_only_loads;
    config.share_host_addresses =
        (Common::TraceMarkers::Experiments3() & Common::TraceMarkers::ShareHostAddresses) != 0;

#if CITRA_ARCH(arm64) && defined(__linux__)
    config.translate_directly =
        (Common::TraceMarkers::Experiments4() & Common::TraceMarkers::LegacyTranslation) == 0;
    config.direct_translation_without =
        (Common::TraceMarkers::Experiments4() & Common::TraceMarkers::TranslationWithout) / 256;
    if (AheadTranslation* const ahead = system.GetAheadTranslation();
        ahead && config.translate_directly) {
        config.translated_ahead = &ahead->Translated();
    }
#endif

    // Multi-process state
    config.processor_id = GetID();
    config.global_monitor = &exclusive_monitor.monitor;

    return std::make_unique<Dynarmic::A32::Jit>(config);
}

} // namespace Core
