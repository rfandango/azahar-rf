// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.
//
// The instruction generator follows Dynarmic's own fuzz test
// (externals/dynarmic/tests/A32, 0BSD, Copyright (c) 2016 MerryMage).

// jitfuzz: the recompiler tested against itself.
//
// Random ARM programs are run through two recompilers that differ only in how
// they are set up (which optimizations, which way of keeping the guest's
// registers and flags), each from the same starting state and with memory of
// its own, and everything the guest can see is compared at the end: registers,
// flags, floating point state, memory written, calls made.
//
// A program ends in a way that does not depend on how the code was cut into
// blocks: by running into a word that is undefined in ARM state (everything
// around the program is that word), which raises an exception at a known place.
// Programs that are still running when their time is up are not compared (their
// end depends on where the recompiler looks at the time), only counted.
//
//   jitfuzz MODE COUNT SEED REFERENCE TEST [LENGTH]
//
// MODE: cb (every memory access is a call), pt (a table of pages looked up in
// the code, calls for the rest), fm (accesses through the host's MMU with a
// signal for the rest). REFERENCE and TEST are words of switches, see
// ApplySwitches.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mcl/stdint.hpp>

#if defined(__aarch64__)
#include "dynarmic/backend/arm64/a32_direct_statistics.h"
#endif
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/frontend/A32/disassembler/disassembler.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/A32/coprocessor.h"
#if defined(__aarch64__)
#include "dynarmic/interface/A32/translated_ahead.h"
#endif
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"
#include "fuzz_util.h"
#include "rand_int.h"

namespace {

using namespace Dynarmic;

constexpr u32 CODE_BASE = 0x00100000;
constexpr u32 DATA_BASE = 0x10000000;
constexpr u32 DATA_SIZE = 0x4000;
/// Undefined in ARM state (the run stops there), a branch to itself in Thumb
/// state
constexpr u32 STOP_WORD = 0xE7FEE7FE;
u64 TIME_GIVEN = 3000;

/// faults=N: how often an access faults before its block is made again the slow
/// way (the emulator's 8 unless given; 1 has every fault remake its block)
u32 g_faults_before_recompile = 8;

/// corpus=FILE: not made-up programs but a game's own code (lines "w ADDRESS WORD" and
/// "e ADDRESS", hexadecimal), entered at one of the addresses with made-up registers and memory
struct Corpus {
  std::unordered_map<u32, u32> words;
  std::vector<u32> entries;
  /// Filled anew for each program
  std::vector<u8> data;
  bool Has(u32 vaddr) const { return words.count(vaddr & ~3u) != 0; }
};
Corpus g_corpus;

/// HostMmuFaults is how the emulator runs once it has pages protected: the page
/// table and the host's view are both given, and some pages cannot be written,
/// or reached at all, where they are. An access that faults there is carried
/// out by the callbacks, and its block is made again with a look into the page
/// table for that access.
enum class Mode { Callbacks, PageTable, HostMmu, HostMmuFaults };

// ------------------------------------------------------------------------------------------
// What the guest runs in

class Env final : public A32::UserCallbacks {
public:
  A32::Jit *jit = nullptr;
  std::vector<u32> code;
  std::map<u32, u8> written;
  /// The pages the recompiled code reaches by itself (page table, host MMU), or
  /// nullptr
  u8 *data = nullptr;
  /// A page behind them that the code may run into, see MakeMachine
  bool spare_page = false;

  u64 ticks_left = 0;
  /// Not zero: the recompiler is given at most this much time at once, so that
  /// a program is left and come back to many times (as the emulator does from
  /// one event to the next)
  u64 slice = 0;
  bool stopped = false;
  u32 stopped_at = 0;
  u32 stop_kind = 0;
  u32 calls = 0;
  u32 call_sum = 0;
  /// What the coprocessors hold and were told
  std::array<u32, 16> coprocessor_words{};
  u32 coprocessor_sum = 0;

  void Begin(const std::vector<u32> &program) {
    for (u32 i = 0; i < coprocessor_words.size(); i++) {
      coprocessor_words[i] = 0x1F000000 + i * 0x01010101;
    }
    coprocessor_sum = 0;
    code = program;
    written.clear();
    if (data) {
      if (!g_corpus.data.empty()) {
        std::memcpy(data, g_corpus.data.data(), DATA_SIZE);
      } else {
        for (u32 i = 0; i < DATA_SIZE; i++) {
          data[i] = static_cast<u8>(i * 7 + (i >> 8));
        }
      }
      if (spare_page) {
        std::memset(data + DATA_SIZE, 0, 4096);
      }
    }
    ticks_left = TIME_GIVEN;
    stopped = false;
    stopped_at = 0;
    stop_kind = 0;
    calls = 0;
    call_sum = 0;
  }

  bool InCode(u32 vaddr) const {
    if (!g_corpus.words.empty()) {
      return g_corpus.Has(vaddr);
    }
    return vaddr >= CODE_BASE && vaddr - CODE_BASE < code.size() * sizeof(u32);
  }
  /// The page behind the data counts where there is one: the code reaches it by
  /// itself, and what goes the long way has to find the same bytes there
  bool InData(u32 vaddr) const {
    return data && vaddr >= DATA_BASE &&
           vaddr - DATA_BASE < DATA_SIZE + (spare_page ? 4096u : 0u);
  }

  /// The program itself is memory nothing writes: a write there goes to
  /// `written`, and a read of it gets the program's words whatever was written
  /// (see MemoryRead8). Said only to the side that is set up to make constants
  /// of such loads.
  bool fold_read_only = false;
  /// How often that was said: so that a run can show the constants were in fact
  /// made
  static inline u64 said_read_only = 0;
  bool IsReadOnlyMemory(u32 vaddr) override {
    const bool read_only = fold_read_only && InCode(vaddr) && InCode(vaddr + 3);
    said_read_only += read_only;
    return read_only;
  }

  std::optional<std::uint32_t> MemoryReadCode(u32 vaddr) override {
    if (!g_corpus.words.empty()) {
      const auto found = g_corpus.words.find(vaddr & ~3u);
      return found != g_corpus.words.end() ? found->second : STOP_WORD;
    }
    if (InCode(vaddr) && InCode(vaddr + 3)) {
      u32 value;
      std::memcpy(
          &value,
          reinterpret_cast<const u8 *>(code.data()) + (vaddr - CODE_BASE), 4);
      return value;
    }
    return STOP_WORD;
  }

  std::uint8_t MemoryRead8(u32 vaddr) override {
    if (InData(vaddr)) {
      return data[vaddr - DATA_BASE];
    }
    if (InCode(vaddr)) {
      if (!g_corpus.words.empty()) {
        return static_cast<u8>(g_corpus.words.at(vaddr & ~3u) >>
                               ((vaddr & 3) * 8));
      }
      return reinterpret_cast<const u8 *>(code.data())[vaddr - CODE_BASE];
    }
    if (const auto iter = written.find(vaddr); iter != written.end()) {
      return iter->second;
    }
    return static_cast<u8>(vaddr ^ (vaddr >> 9));
  }
  std::uint16_t MemoryRead16(u32 vaddr) override {
    return u16(MemoryRead8(vaddr)) | u16(MemoryRead8(vaddr + 1)) << 8;
  }
  std::uint32_t MemoryRead32(u32 vaddr) override {
    return u32(MemoryRead16(vaddr)) | u32(MemoryRead16(vaddr + 2)) << 16;
  }
  std::uint64_t MemoryRead64(u32 vaddr) override {
    return u64(MemoryRead32(vaddr)) | u64(MemoryRead32(vaddr + 4)) << 32;
  }

  void MemoryWrite8(u32 vaddr, std::uint8_t value) override {
    if (InData(vaddr)) {
      data[vaddr - DATA_BASE] = value;
      return;
    }
    written[vaddr] = value;
  }
  void MemoryWrite16(u32 vaddr, std::uint16_t value) override {
    MemoryWrite8(vaddr, static_cast<u8>(value));
    MemoryWrite8(vaddr + 1, static_cast<u8>(value >> 8));
  }
  void MemoryWrite32(u32 vaddr, std::uint32_t value) override {
    MemoryWrite16(vaddr, static_cast<u16>(value));
    MemoryWrite16(vaddr + 2, static_cast<u16>(value >> 16));
  }
  void MemoryWrite64(u32 vaddr, std::uint64_t value) override {
    MemoryWrite32(vaddr, static_cast<u32>(value));
    MemoryWrite32(vaddr + 4, static_cast<u32>(value >> 32));
  }

  // (One processor: nothing comes in between)
  bool MemoryWriteExclusive8(u32 vaddr, std::uint8_t value,
                             std::uint8_t) override {
    MemoryWrite8(vaddr, value);
    return true;
  }
  bool MemoryWriteExclusive16(u32 vaddr, std::uint16_t value,
                              std::uint16_t) override {
    MemoryWrite16(vaddr, value);
    return true;
  }
  bool MemoryWriteExclusive32(u32 vaddr, std::uint32_t value,
                              std::uint32_t) override {
    MemoryWrite32(vaddr, value);
    return true;
  }
  bool MemoryWriteExclusive64(u32 vaddr, std::uint64_t value,
                              std::uint64_t) override {
    MemoryWrite64(vaddr, value);
    return true;
  }

  void InterpreterFallback(u32 pc, size_t) override { Stop(pc, 0x100); }

  /// What the emulated kernel does in Azahar: reads the registers, writes some
  /// of them
  void CallSVC(std::uint32_t swi) override {
    auto &regs = jit->Regs();
    u32 sum = swi * 0x9E3779B1u + jit->Cpsr();
    for (int i = 0; i < 15; i++) {
      sum = (sum << 5 | sum >> 27) ^ regs[i];
    }
    calls++;
    call_sum = call_sum * 31 + sum;
    regs[0] ^= sum;
    regs[1] = sum * 3;
    regs[2] += swi;
    regs[7] ^= 0x00FF00FF;
  }

  void ExceptionRaised(u32 pc, A32::Exception exception) override {
    Stop(pc, 1 + static_cast<u32>(exception));
  }

  void AddTicks(std::uint64_t ticks) override {
    ticks_left = ticks > ticks_left ? 0 : ticks_left - ticks;
  }
  std::uint64_t GetTicksRemaining() override {
    if (slice == 0) {
      return ticks_left;
    }
    // Lengths that change, the same for both sides: from the time that is left
    const u64 now = 1 + (ticks_left * 2654435761u >> 7) % slice;
    return std::min(now, ticks_left);
  }

private:
  void Stop(u32 pc, u32 kind) {
    if (!stopped) {
      stopped = true;
      stopped_at = pc;
      stop_kind = kind;
    }
    jit->HaltExecution();
  }
};

/// A coprocessor that answers everything: by a call for some of its registers,
/// by a place in memory the code reads and writes itself for the others (the
/// emulator's own does both)
class TestCoprocessor final : public A32::Coprocessor {
public:
  explicit TestCoprocessor(Env &env) : env{env} {}

  std::optional<Callback>
  CompileInternalOperation(bool, unsigned, A32::CoprocReg, A32::CoprocReg,
                           A32::CoprocReg, unsigned) override {
    return Callback{&Asked, &env};
  }
  CallbackOrAccessOneWord CompileSendOneWord(bool, unsigned, A32::CoprocReg CRn,
                                             A32::CoprocReg CRm,
                                             unsigned) override {
    if (static_cast<unsigned>(CRm) & 1) {
      return &env.coprocessor_words[static_cast<size_t>(CRn) & 15];
    }
    return Callback{&ToldOne, &env};
  }
  CallbackOrAccessTwoWords CompileSendTwoWords(bool, unsigned,
                                               A32::CoprocReg CRm) override {
    if (static_cast<unsigned>(CRm) & 1) {
      return std::array<std::uint32_t *, 2>{&env.coprocessor_words[2],
                                            &env.coprocessor_words[3]};
    }
    return Callback{&Told, &env};
  }
  CallbackOrAccessOneWord CompileGetOneWord(bool, unsigned, A32::CoprocReg CRn,
                                            A32::CoprocReg CRm,
                                            unsigned) override {
    if (static_cast<unsigned>(CRm) & 1) {
      return &env.coprocessor_words[static_cast<size_t>(CRn) & 15];
    }
    return Callback{&Asked, &env};
  }
  CallbackOrAccessTwoWords CompileGetTwoWords(bool, unsigned,
                                              A32::CoprocReg CRm) override {
    if (static_cast<unsigned>(CRm) & 1) {
      return std::array<std::uint32_t *, 2>{&env.coprocessor_words[4],
                                            &env.coprocessor_words[5]};
    }
    return Callback{&Asked, &env};
  }
  std::optional<Callback>
  CompileLoadWords(bool, bool, A32::CoprocReg,
                   std::optional<std::uint8_t>) override {
    return Callback{&ToldOne, &env};
  }
  std::optional<Callback>
  CompileStoreWords(bool, bool, A32::CoprocReg,
                    std::optional<std::uint8_t>) override {
    return Callback{&ToldOne, &env};
  }

private:
  // Only what a callback of its kind is given counts: the rest is whatever the
  // registers held
  static std::uint64_t Told(void *arg, std::uint32_t a, std::uint32_t b) {
    Env &env = *static_cast<Env *>(arg);
    env.coprocessor_sum = env.coprocessor_sum * 33 + a + b * 7 + 1;
    return 0;
  }
  static std::uint64_t ToldOne(void *arg, std::uint32_t a, std::uint32_t) {
    Env &env = *static_cast<Env *>(arg);
    env.coprocessor_sum = env.coprocessor_sum * 33 + a + 2;
    return 0;
  }
  static std::uint64_t Asked(void *arg, std::uint32_t, std::uint32_t) {
    Env &env = *static_cast<Env *>(arg);
    env.coprocessor_sum = env.coprocessor_sum * 33 + 5;
    return (std::uint64_t{env.coprocessor_sum ^ 0xA5A5A5A5} << 32) |
           (env.coprocessor_sum * 0x9E3779B1u);
  }

  Env &env;
};

#if defined(__aarch64__)
class AheadMemory;
#endif

struct Machine {
  std::unique_ptr<Env> env;
  std::unique_ptr<
      std::array<std::uint8_t *, A32::UserConfig::NUM_PAGE_TABLE_ENTRIES>>
      pages;
  std::vector<u8> access;
  u8 *view = nullptr;
  std::unique_ptr<ExclusiveMonitor> monitor;
#if defined(__aarch64__)
  std::unique_ptr<A32::TranslatedAhead> ahead;
  std::unique_ptr<AheadMemory> ahead_memory;
#endif
  std::unique_ptr<A32::Jit> jit;
};

/// The switches of a configuration word. Bits 0 to 3 are what the recompiler
/// always had; the rest name the ways of generating code that are being tested.
void ApplySwitches(A32::UserConfig &config, u32 switches) {
  if (switches & 1) {
    config.optimizations &= ~OptimizationFlag::FastDispatch;
  }
  if (switches & 2) {
    config.optimizations = no_optimizations;
  }
  if (switches & 4) {
    config.optimizations &= ~OptimizationFlag::BlockLinking;
  }
  if (switches & 8) {
    config.optimizations &= ~OptimizationFlag::ReturnStackBuffer;
  }
  if (switches & 16) {
    config.pin_guest_registers = true;
  }
  if (switches & 32) {
    config.write_into_guest_registers = true;
  }
  if (switches & 64) {
    config.fold_conditional_branches = true;
  }
  if (switches & 128) {
    config.guest_flags_in_host_flags = true;
  }
  if (switches & 256) {
    config.fp_status_in_host = true;
  }
  if (switches & 512) {
    config.fp_registers_through_general = true;
  }
  if (switches & 1024) {
    config.look_up_blocks_in_place = true;
  }
  if (switches & 2048) {
    config.pin_guest_fp_registers = true;
  }
  if (switches & 4096) {
    config.conditional_instructions_in_blocks = true;
  }
  // (8192 and 16384 were the longer blocks, which are gone)
  if (switches & 32768) {
    config.fold_read_only_loads = true;
  }
  if (switches & 65536) {
    config.share_host_addresses = true;
  }
  if (switches & 131072) {
    config.translate_directly = true;
  }
  // (262144: translated ahead, see MakeMachine)
  if (switches & 524288) {
    config.tables_of_code_by_address = false;
  }
  config.direct_translation_without = switches >> 20 & 31;
}

/// The guest's address space in the host's, empty: at a multiple of 4 GiB, with
/// room on either side that nothing is in
constexpr u64 GUARD_SIZE = u64{8} << 30;

u8 *MakeView(A32::UserConfig &config) {
  const size_t size = (u64{4} << 30) * 2 + GUARD_SIZE * 2;
  u8 *const reserved = static_cast<u8 *>(
      mmap(nullptr, size, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
  if (reserved == MAP_FAILED) {
    std::perror("mmap of the guest's address space");
    std::exit(2);
  }
  const u64 aligned =
      (reinterpret_cast<u64>(reserved) + GUARD_SIZE + 0xFFFFFFFF) &
      ~u64{0xFFFFFFFF};
  config.fastmem_guard_size = GUARD_SIZE;
  return reinterpret_cast<u8 *>(aligned);
}

#if defined(__aarch64__)
/// 262144: the code is translated ahead, from a copy
class AheadMemory final : public A32::TranslatedAhead::Memory {
public:
  Env &env;
  std::vector<u32> code;
  bool read_only = false;

  explicit AheadMemory(Env &env) : env{env} {}
  std::optional<std::uint32_t> Read(std::uint32_t vaddr) override {
    if (!g_corpus.words.empty()) {
      const auto found = g_corpus.words.find(vaddr & ~3u);
      return found != g_corpus.words.end()
                 ? std::optional<std::uint32_t>{found->second}
                 : std::nullopt;
    }
    if (vaddr < CODE_BASE || (vaddr - CODE_BASE) / 4 >= code.size()) {
      return std::nullopt;
    }
    return code[(vaddr - CODE_BASE) / 4];
  }
  bool IsReadOnly(std::uint32_t vaddr) override {
    return read_only && Read(vaddr & ~3u).has_value();
  }
  std::uint64_t GetTicksForCode(std::uint32_t vaddr,
                                std::uint32_t instruction) override {
    return env.GetTicksForCode(false, vaddr, instruction);
  }
};
#endif

Machine MakeMachine(Mode mode, u32 switches) {
  Machine m;
  m.env = std::make_unique<Env>();
  m.env->fold_read_only = (switches & 32768) != 0;
  A32::UserConfig config;
  config.callbacks = m.env.get();
  m.monitor = std::make_unique<ExclusiveMonitor>(1);
  config.global_monitor = m.monitor.get();
  config.define_unpredictable_behaviour = true;
  config.fastmem_faults_before_recompile = g_faults_before_recompile;
  config.direct_faults_before_check = g_faults_before_recompile;
  // A program is a few dozen instructions, and the room for code is emptied
  // before each one
  config.code_cache_size = (g_corpus.words.empty() ? 8 : 100) * 1024 * 1024;
  for (auto &coprocessor : config.coprocessors) {
    coprocessor = std::make_shared<TestCoprocessor>(*m.env);
  }
  if (mode == Mode::PageTable) {
    m.pages = std::make_unique<
        std::array<std::uint8_t *, A32::UserConfig::NUM_PAGE_TABLE_ENTRIES>>();
    m.pages->fill(nullptr);
    // One page more than the table knows: an access that begins in the last
    // page and is not aligned runs past its end (the code looks the page of its
    // first byte up, as on the emulator, where the pages of a region lie one
    // after the other)
    m.env->data = static_cast<u8 *>(mmap(nullptr, DATA_SIZE + 4096,
                                         PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    m.env->spare_page = true;
    for (u32 i = 0; i < DATA_SIZE / 4096; i++) {
      (*m.pages)[(DATA_BASE >> 12) + i] = m.env->data + i * 4096;
    }
    config.page_table = m.pages.get();
  } else if (mode == Mode::HostMmu) {
    m.view = MakeView(config);
    mprotect(m.view + DATA_BASE, DATA_SIZE, PROT_READ | PROT_WRITE);
    m.env->data = m.view + DATA_BASE;
    config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(m.view);
  } else if (mode == Mode::HostMmuFaults) {
    m.view = MakeView(config);
    // The same bytes twice: in the view, with what the code may do there, and
    // where the callbacks and the page table reach all of them. One page more,
    // as for the page table.
    const int fd = static_cast<int>(syscall(SYS_memfd_create, "jitfuzz", 0u));
    if (m.view == MAP_FAILED || fd < 0 ||
        ftruncate(fd, DATA_SIZE + 4096) != 0) {
      std::perror("memory of the guest");
      std::exit(2);
    }
    m.env->data = static_cast<u8 *>(mmap(
        nullptr, DATA_SIZE + 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    u8 *const in_view = static_cast<u8 *>(
        mmap(m.view + DATA_BASE, DATA_SIZE + 4096, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_FIXED, fd, 0));
    close(fd);
    if (m.env->data == MAP_FAILED || in_view == MAP_FAILED) {
      std::perror("mapping the memory of the guest");
      std::exit(2);
    }
    m.env->spare_page = true;
    m.pages = std::make_unique<
        std::array<std::uint8_t *, A32::UserConfig::NUM_PAGE_TABLE_ENTRIES>>();
    m.pages->fill(nullptr);
    m.access.resize(A32::UserConfig::NUM_PAGE_TABLE_ENTRIES);
    config.page_access = m.access.data();
    for (u32 i = 0; i < DATA_SIZE / 4096; i++) {
      // Page 1 is read where it is and written through the callbacks (a page
      // the emulator watches), page 2 is not reached at all (one it has shut)
      const bool direct = i != 1 && i != 2;
      mprotect(m.view + DATA_BASE + i * 4096, 4096,
               direct   ? PROT_READ | PROT_WRITE
               : i == 1 ? PROT_READ
                        : PROT_NONE);
      (*m.pages)[(DATA_BASE >> 12) + i] =
          direct ? m.env->data + i * 4096 : nullptr;
      m.access[(DATA_BASE >> 12) + i] = direct ? 3 : i == 1 ? 1 : 0;
    }
    config.page_table = m.pages.get();
    config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(m.view);
  }
  ApplySwitches(config, switches);
#if defined(__aarch64__)
  if (switches & 262144) {
    m.ahead = std::make_unique<A32::TranslatedAhead>();
    m.ahead_memory = std::make_unique<AheadMemory>(*m.env);
    m.ahead_memory->read_only = m.env->fold_read_only;
    config.translated_ahead = m.ahead.get();
  }
#endif
  m.jit = std::make_unique<A32::Jit>(config);
  m.env->jit = m.jit.get();
  return m;
}

// ------------------------------------------------------------------------------------------
// Programs

u32 Bits(u32 max) { return RandInt<u32>(0, max); }
bool Chance(u32 percent) { return Bits(99) < percent; }

/// Whether an instruction can stand in the middle of a program: it goes on to
/// the next one, is nothing the test leaves out, and writes none of the
/// registers in `keep`.
bool Usable(u32 instruction, u32 pc, u32 keep) {
  const A32::LocationDescriptor location{pc, {}, {}};
  IR::Block block{location};
  if (!A32::TranslateSingleInstruction(block, location, instruction)) {
    return false;
  }
  for (const auto &inst : block) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::A32ExceptionRaised:
    case IR::Opcode::A32CallSupervisor:
    case IR::Opcode::A32BXWritePC:
      return false;
    case IR::Opcode::A32SetRegister:
      if (keep >> static_cast<u32>(inst.GetArg(0).GetA32RegRef()) & 1) {
        return false;
      }
      break;
    default:
      break;
    }
  }
  return true;
}

struct Generators {
  std::vector<InstructionGenerator> list;
};

const Generators &AllGenerators(bool with_simd) {
  static const auto make = [](bool simd) {
    std::vector<std::tuple<std::string, const char *>> list{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/arm.inc"
#include "dynarmic/frontend/A32/decoder/vfp.inc"
#undef INST
    };
    if (simd) {
      const std::vector<std::tuple<std::string, const char *>> more{
#define INST(fn, name, bitstring) {#fn, bitstring},
#include "dynarmic/frontend/A32/decoder/asimd.inc"
#undef INST
      };
      list.insert(list.end(), more.begin(), more.end());
    }
    static constexpr std::array leave_out{
        // As another user would do them
        "arm_LDRBT",
        "arm_LDRHT",
        "arm_LDRSBT",
        "arm_LDRSHT",
        "arm_LDRT",
        "arm_STRBT",
        "arm_STRHT",
        "arm_STRT",
        // Need the monitor shared between processors
        "arm_LDREXB",
        "arm_LDREXD",
        "arm_LDREXH",
        "arm_LDREX",
        "arm_LDAEXB",
        "arm_LDAEXD",
        "arm_LDAEXH",
        "arm_LDAEX",
        "arm_STREXB",
        "arm_STREXD",
        "arm_STREXH",
        "arm_STREX",
        "arm_STLEXB",
        "arm_STLEXD",
        "arm_STLEXH",
        "arm_STLEX",
        "arm_SWP",
        "arm_SWPB",
        // Fault on an address that is not aligned, on the host as on the guest
        "arm_LDA",
        "arm_LDAB",
        "arm_LDAH",
        "arm_STL",
        "arm_STLB",
        "arm_STLH",
        // Of other modes of the processor
        "arm_LDM_eret",
        "arm_LDM_usr",
        "arm_STM_usr",
        "arm_CPS",
        "arm_RFE",
        "arm_SRS",
        // Change how code is read
        "arm_BLX_reg",
        "arm_BLX_imm",
        "arm_BXJ",
        "arm_SETEND",
        // The translator stops the program at one form of these (base register
        // PC, written back)
        "arm_LDC",
        "arm_STC",
        "arm_UDF",
    };
    Generators result;
    for (const auto &[fn, bitstring] : list) {
      if (std::find_if(leave_out.begin(), leave_out.end(),
                       [&](const char *name) { return fn == name; }) ==
          leave_out.end()) {
        result.list.emplace_back(InstructionGenerator{bitstring});
      }
    }
    return result;
  };
  static const Generators plain = make(false);
  static const Generators simd = make(true);
  return with_simd ? simd : plain;
}

/// modes=0: the floating point mode is left alone
bool g_with_modes = true;
bool g_with_simd = false;

/// A random instruction for the middle of a program
u32 Plain(u32 pc, u32 keep) {
  const Generators &gens = AllGenerators(g_with_simd);
  while (true) {
    const InstructionGenerator &gen =
        gens.list[RandInt<size_t>(0, gens.list.size() - 1)];
    u32 inst = gen.Generate();
    const bool has_condition = (gen.Mask() & 0xF0000000) == 0;
    if (has_condition) {
      if ((inst & 0xF0000000) == 0xF0000000) {
        continue;
      }
      // Half of all instructions are unconditional, as in real code
      if (Chance(50)) {
        inst = (inst & 0x0FFFFFFF) | 0xE0000000;
      }
    }
    if (Usable(inst, pc, keep)) {
      return inst;
    }
  }
}

u32 Condition() { return (Chance(40) ? 0xEu : Bits(13)) << 28; }

/// B or BL to the instruction `words` after the next one
u32 Branch(u32 words_ahead, bool link, u32 condition) {
  return condition | (link ? 0x0B000000 : 0x0A000000) |
         ((words_ahead - 1) & 0x00FFFFFF);
}

/// A slot in the middle: mostly a random instruction, now and then a short jump
/// ahead or a call of the kernel
u32 Slot(u32 pc, u32 keep, u32 left_after) {
  if (left_after >= 1 && Chance(9)) {
    return Branch(Bits(std::min<u32>(left_after, 4)), false, Condition());
  }
  if (Chance(3)) {
    return Condition() | 0x0F000000 | Bits(0xFF);
  }
  if (Chance(6)) {
    // A word loaded from next to the code, at a distance from the program
    // counter, as compilers load their constants: into a register (ldr rX, [pc,
    // #n]) or as a float (vldr sX, [pc, #n])
    const u32 words = Bits(15);
    const u32 up = Bits(1) << 23;
    if (Chance(70)) {
      u32 rd = Bits(10);
      while ((keep >> rd) & 1) {
        rd = Bits(10);
      }
      return Condition() | 0x051F0000 | up | rd << 12 | words * 4;
    }
    return Condition() | 0x0D1F0A00 | up | Bits(1) << 22 | Bits(15) << 12 |
           words;
  }
  if (Chance(8)) {
    // A load or store at a fixed distance ahead of a register, as the fields of
    // a structure and the slots of the stack are reached: by the stack pointer
    // more often than not, so that a block has several by the same register.
    // Words, bytes and halves; distances that are multiples of four mostly,
    // small odd ones and far ones now and then.
    const u32 base = Chance(60) ? 13 : Bits(12);
    u32 rd = Bits(10);
    while ((keep >> rd) & 1) {
      rd = Bits(10);
    }
    const u32 kind = Bits(5);
    const u32 distance = kind >= 4    ? (Chance(70) ? Bits(127) * 2 : Bits(255))
                         : Chance(75) ? Bits(63) * 4
                         : Chance(50) ? Bits(255)
                                      : Bits(1022) * 4;
    static constexpr std::array<u32, 4> word_or_byte{
        0x05900000, 0x05800000, 0x05D00000, 0x05C00000}; // ldr, str, ldrb, strb
    if (kind < 4) {
      return Condition() | word_or_byte[kind] | base << 16 | rd << 12 |
             distance;
    }
    // ldrh, strh
    return Condition() | (kind == 4 ? 0x01D000B0 : 0x01C000B0) | base << 16 |
           rd << 12 | (distance >> 4) << 8 | (distance & 15);
  }
  return Plain(pc, keep);
}

void Fill(std::vector<u32> &program, u32 count, u32 keep) {
  for (u32 i = 0; i < count; i++) {
    // Now and then the floating point mode is read, set, or put back
    if (g_with_modes && Chance(5)) {
      if (Chance(40)) {
        program.push_back(0xEEF16A10); // vmrs r6, fpscr
        continue;
      }
      if (Chance(30)) {
        program.push_back(0xEEE16A10); // vmsr fpscr, r6
        continue;
      }
      if (count - i >= 3) {
        static constexpr std::array<u32, 5> modes{
            0xE3A07403,  // mov r7, #0x03000000
            0xE3A07407,  // #0x07000000
            0xE3A075C0,  // #0x30000000: flags, and nothing else
            0xE3A07000,  // #0
            0xE3A075F0}; // #0x3C000000
        program.push_back(modes[Bits(4)]);
        // orr r7, r7, #LENGTH << 16, or the distance of two besides
        program.push_back(Chance(75) ? 0xE3877800 | Bits(7)
                                     : 0xE3877833 | Bits(7));
        program.push_back(0xEEE17A10); // vmsr fpscr, r7
        i += 2;
        continue;
      }
    }
    // A constant that is an address, and what is done there; or one of four by a register
    if (count - i >= 6 && Chance(5)) {
      const auto free_register = [&] {
        u32 reg = Bits(10);
        while ((keep >> reg) & 1) {
          reg = Bits(10);
        }
        return reg;
      };
      const u32 base = free_register();
      if (Chance(30)) {
        const u32 index = free_register();
        if (Chance(80)) {
          program.push_back(0xE2000003 | index << 16 | index << 12); // and
          i++;
        }
        // ldr base, [pc, index, lsl #2]; b over the words
        program.push_back(0xE79F0100 | base << 12 | index);
        program.push_back(Branch(4, false, 0xE0000000));
        for (u32 n = 0; n < 4; n++) {
          program.push_back(DATA_BASE + Bits(DATA_SIZE - 1));
        }
        i += 5;
      } else {
        // ldr base, [pc]; b over the word
        program.push_back(0xE59F0000 | base << 12);
        program.push_back(Branch(1, false, 0xE0000000));
        program.push_back(DATA_BASE + (Bits(DATA_SIZE - 1) & (Chance(85) ? ~3u : ~0u)));
        i += 2;
      }
      for (u32 n = 1 + Bits(2); n != 0 && i + 1 < count; n--, i++) {
        static constexpr std::array<u32, 4> accesses{
            0x05900000, 0x05800000, 0x05D00000, 0x05C00000}; // ldr, str, ldrb, strb
        program.push_back(Condition() | accesses[Bits(3)] | base << 16 |
                          free_register() << 12 | Bits(63) * 4);
      }
      continue;
    }
    const u32 pc = CODE_BASE + static_cast<u32>(program.size()) * 4;
    program.push_back(Slot(pc, keep, count - 1 - i));
  }
}

/// Any instruction at all, also one that leaves for wherever a register says
u32 Last(u32 pc) {
  const Generators &gens = AllGenerators(g_with_simd);
  while (true) {
    const InstructionGenerator &gen =
        gens.list[RandInt<size_t>(0, gens.list.size() - 1)];
    const u32 inst = gen.Generate();
    if ((gen.Mask() & 0xF0000000) == 0 && (inst & 0xF0000000) == 0xF0000000) {
      continue;
    }
    const A32::LocationDescriptor location{pc, {}, {}};
    IR::Block block{location};
    A32::TranslateSingleInstruction(block, location, inst);
    bool ok = true;
    for (const auto &ir : block) {
      switch (ir.GetOpcode()) {
      case IR::Opcode::A32ExceptionRaised:
      case IR::Opcode::A32CallSupervisor:
        ok = false;
        break;
      default:
        break;
      }
    }
    if (ok) {
      return inst;
    }
  }
}

std::vector<u32> MakeProgram(u32 length) {
  std::vector<u32> program;
  const u32 shape = Bits(9);
  if (shape < 4) {
    // One run of instructions
    Fill(program, length, 0);
  } else if (shape < 7) {
    // A loop that r11 counts
    const u32 body = 2 + Bits(std::max<u32>(length / 2, 3));
    Fill(program, Bits(length / 3), 0);
    program.push_back(0xE3A0B000 | (2 + Bits(3))); // mov r11, #n
    const u32 loop_start = static_cast<u32>(program.size());
    Fill(program, body, 1u << 11);
    program.push_back(0xE25BB001); // subs r11, r11, #1
    const u32 here = static_cast<u32>(program.size());
    program.push_back(0x1A000000 |
                      ((loop_start - here - 2) & 0x00FFFFFF)); // bne loop
    Fill(program, Bits(length / 3), 0);
  } else {
    // A routine that is called once or twice and returns
    const bool with_stack = Chance(50);
    const u32 keep = Chance(80) ? (1u << 14) | (with_stack ? 1u << 13 : 0) : 0;
    const u32 routine = 2 + Bits(std::max<u32>(length / 3, 3));
    const u32 before = Bits(length / 4);
    const u32 between = Bits(length / 4);
    const bool twice = Chance(40);
    const u32 after = Bits(length / 4);
    Fill(program, before, 0);
    // The distances to the routine are filled in below
    const u32 first_call = static_cast<u32>(program.size());
    program.push_back(0);
    Fill(program, between, with_stack ? 1u << 13 : 0);
    const u32 second_call = static_cast<u32>(program.size());
    if (twice) {
      program.push_back(0);
    }
    Fill(program, after, 0);
    const u32 skip = static_cast<u32>(program.size());
    program.push_back(0);
    const u32 routine_start = static_cast<u32>(program.size());
    if (with_stack) {
      program.push_back(0xE92D4010); // push {r4, lr}
    }
    Fill(program, routine, keep);
    static constexpr std::array<u32, 2> returns{
        0xE12FFF1E, 0xE1A0F00E}; // bx lr; mov pc, lr
    program.push_back(with_stack ? 0xE8BD8010
                                 : returns[Bits(1)]); // pop {r4, pc}
    const u32 end = static_cast<u32>(program.size());
    const u32 call_condition = Chance(70) ? 0xE0000000 : Condition();
    program[first_call] =
        Branch(routine_start - first_call - 1, true, call_condition);
    if (twice) {
      program[second_call] =
          Branch(routine_start - second_call - 1, true, Condition());
    }
    program[skip] = Branch(end - skip - 1, false, 0xE0000000);
    Fill(program, 1 + Bits(3), 0);
  }
  if (Chance(35)) {
    program.push_back(Last(CODE_BASE + static_cast<u32>(program.size()) * 4));
  }
  return program;
}

struct Start {
  std::array<u32, 16> regs;
  std::array<u32, 64> ext_regs;
  u32 cpsr;
  u32 fpscr;
};

u32 InterestingValue() {
  switch (Bits(7)) {
  case 0:
    return DATA_BASE + (Bits(DATA_SIZE - 1) & ~3u);
  case 1:
    return DATA_BASE + Bits(DATA_SIZE + 64) - 32;
  case 2:
    return Bits(40);
  case 3:
    return 0u - Bits(40);
  case 4:
    return Chance(50) ? 0x80000000 : 0x7FFFFFFF;
  default:
    return RandInt<u32>(0, 0xFFFFFFFF);
  }
}

u32 InterestingFloat() {
  switch (Bits(9)) {
  case 0:
    return 0;
  case 1:
    return 0x80000000;
  case 2:
    return 0x7F800000;
  case 3:
    return 0x7FC00000 | Bits(0xFFFF);
  case 4:
    return 0x7F800001 + Bits(0xFFFF);
  case 5:
    return Bits(0x7FFFFF);
  case 6:
    return 0x3F800000;
  default:
    return RandInt<u32>(0, 0xFFFFFFFF);
  }
}

Start MakeStart() {
  Start s;
  for (u32 &r : s.regs) {
    r = InterestingValue();
  }
  // A stack in the pages the code reaches by itself, more often than not
  if (Chance(70)) {
    s.regs[13] = DATA_BASE + DATA_SIZE / 2 + (Bits(0x400) & ~3u);
  }
  s.regs[15] = CODE_BASE;
  for (u32 &r : s.ext_regs) {
    r = InterestingFloat();
  }
  // User mode, ARM state, the four flags, the sticky flag and the four "greater
  // or equal"
  s.cpsr = 0x10 | (Bits(0xF) << 28) | (Bits(1) << 27) | (Bits(0xF) << 16);
  // Rounding mode, flush to zero, default NaN; now and then vectors of the old
  // kind, also such as there are not
  s.fpscr = Bits(3) << 22 | Bits(1) << 24 | Bits(1) << 25;
  if (g_with_modes && Chance(25)) {
    s.fpscr |= Bits(7) << 16 | (Chance(25) ? 3u << 20 : 0);
  }
  return s;
}

void LoadCorpus(const char *path) {
  std::FILE *const file = std::fopen(path, "r");
  if (!file) {
    std::perror(path);
    std::exit(2);
  }
  char line[128];
  while (std::fgets(line, sizeof(line), file)) {
    unsigned a = 0, b = 0;
    if (std::sscanf(line, "w %x %x", &a, &b) == 2) {
      g_corpus.words[a] = b;
    } else if (std::sscanf(line, "e %x", &a) == 1) {
      g_corpus.entries.push_back(a);
    }
  }
  std::fclose(file);
  TIME_GIVEN = 8000;
}

u32 SomeFloat() {
  const float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 1.0f / 3, 100.0f};
  float value = Chance(40) ? values[Bits(6)]
                           : static_cast<float>(static_cast<s32>(Bits(20000)) -
                                                10000) /
                                 (1 + Bits(99));
  u32 word;
  std::memcpy(&word, &value, 4);
  return word;
}

u32 SomePointer() { return DATA_BASE + 0x400 + (Bits(0x2B00) & ~15u); }

/// A word as a game's memory and registers have them
u32 SomeWord() {
  const u32 kind = Bits(99);
  return kind < 45   ? SomePointer()
         : kind < 60 ? Bits(16)
         : kind < 70 ? Bits(1)
         : kind < 90 ? SomeFloat()
                     : RandInt<u32>(0, 0xFFFFFFFF);
}

/// Registers for a game's code, and memory
Start MakeCorpusStart() {
  g_corpus.data.resize(DATA_SIZE);
  for (u32 i = 0; i < DATA_SIZE; i += 4) {
    const u32 word = SomeWord();
    std::memcpy(&g_corpus.data[i], &word, 4);
  }
  Start s;
  for (u32 &r : s.regs) {
    r = Chance(70) ? SomePointer() : SomeWord();
  }
  s.regs[13] = DATA_BASE + DATA_SIZE - 0x800;
  s.regs[14] = 0x00000100; // nothing there: the program stops
  s.regs[15] = g_corpus.entries[RandInt<size_t>(0, g_corpus.entries.size() - 1)];
  for (u32 &r : s.ext_regs) {
    r = SomeFloat();
  }
  s.cpsr = 0x10 | (Bits(0xF) << 28);
  s.fpscr = Chance(75) ? 0x03000000 : Bits(3) << 22 | Bits(1) << 24 | Bits(1) << 25;
  return s;
}

// ------------------------------------------------------------------------------------------
// Running and comparing

struct Result {
  std::array<u32, 16> regs;
  std::array<u32, 64> ext_regs;
  u32 cpsr;
  u32 fpscr;
  bool stopped;
  u32 stopped_at;
  u32 stop_kind;
  u32 calls;
  u32 call_sum;
  std::array<u32, 16> coprocessor_words;
  u32 coprocessor_sum;
  u64 ticks_left;
  std::map<u32, u8> written;
  std::vector<u8> data;
};

#if defined(__aarch64__)
/// Some of the program is translated ahead, now and then from a copy that is not what will be in
/// memory, and what was translated of the program before is mostly still there: code of something
/// else at the same addresses, which must not be taken
void TranslateAhead(Machine &m, const std::vector<u32> &program,
                    const Start &start) {
  if (Chance(30)) {
    m.ahead->Clear();
  }
  m.ahead_memory->code = program;
  if (Chance(20) && !program.empty()) {
    m.ahead_memory->code[Bits(static_cast<u32>(program.size()) - 1)] ^=
        1u << Bits(27);
  }
  const u32 times = Bits(6);
  for (u32 i = 0; i < times; i++) {
    const u32 pc =
        i == 0 && Chance(70)
            ? start.regs[15]
            : CODE_BASE + 4 * Bits(static_cast<u32>(program.size()) - 1);
    const A32::LocationDescriptor location{pc & ~3u, A32::PSR{start.cpsr},
                                           A32::FPSCR{start.fpscr}};
    m.ahead->Translate(location.UniqueHash(), *m.ahead_memory);
  }
}
#endif

Result RunOn(Machine &m, const std::vector<u32> &program, const Start &start,
             u32 spoil) {
  m.env->Begin(program);
#if defined(__aarch64__)
  if (m.ahead && g_corpus.words.empty() && Chance(60)) {
    // The program has changed, and that is all the recompiler is told
    m.jit->InvalidateCacheRange(CODE_BASE, 0x8000);
  } else
#endif
  if (g_corpus.words.empty()) {
    m.jit->ClearCache();
  }
  m.jit->Reset();
#if defined(__aarch64__)
  if (m.ahead && g_corpus.words.empty()) {
    TranslateAhead(m, program, start);
  }
#endif
  m.jit->Regs() = start.regs;
  m.jit->ExtRegs() = start.ext_regs;
  m.jit->SetCpsr(start.cpsr);
  m.jit->SetFpscr(start.fpscr);
  if (spoil) {
    m.jit->Regs()[5] ^= spoil;
  }
  while (!m.env->stopped && m.env->ticks_left > 0) {
    m.jit->Run();
  }
  Result r;
  r.regs = m.jit->Regs();
  r.ext_regs = m.jit->ExtRegs();
  r.cpsr = m.jit->Cpsr();
  r.fpscr = m.jit->Fpscr();
  r.stopped = m.env->stopped;
  r.stopped_at = m.env->stopped_at;
  r.stop_kind = m.env->stop_kind;
  r.calls = m.env->calls;
  r.call_sum = m.env->call_sum;
  r.coprocessor_words = m.env->coprocessor_words;
  r.coprocessor_sum = m.env->coprocessor_sum;
  r.ticks_left = m.env->ticks_left;
  r.written = m.env->written;
  if (m.env->data) {
    r.data.assign(m.env->data, m.env->data + DATA_SIZE);
  }
  return r;
}

std::string Difference(const Result &a, const Result &b) {
  std::string out;
  char line[160];
  for (int i = 0; i < 16; i++) {
    if (a.regs[i] != b.regs[i]) {
      std::snprintf(line, sizeof(line), "  r%d: %08x against %08x\n", i,
                    a.regs[i], b.regs[i]);
      out += line;
    }
  }
  if (a.cpsr != b.cpsr) {
    std::snprintf(line, sizeof(line), "  cpsr: %08x against %08x\n", a.cpsr,
                  b.cpsr);
    out += line;
  }
  if (a.fpscr != b.fpscr) {
    std::snprintf(line, sizeof(line), "  fpscr: %08x against %08x\n", a.fpscr,
                  b.fpscr);
    out += line;
  }
  for (int i = 0; i < 64; i++) {
    if (a.ext_regs[i] != b.ext_regs[i]) {
      std::snprintf(line, sizeof(line), "  s%d: %08x against %08x\n", i,
                    a.ext_regs[i], b.ext_regs[i]);
      out += line;
    }
  }
  if (a.stopped_at != b.stopped_at || a.stop_kind != b.stop_kind) {
    std::snprintf(line, sizeof(line),
                  "  stopped at %08x (kind %u) against %08x (kind %u)\n",
                  a.stopped_at, a.stop_kind, b.stopped_at, b.stop_kind);
    out += line;
  }
  if (a.calls != b.calls || a.call_sum != b.call_sum) {
    std::snprintf(line, sizeof(line),
                  "  kernel calls: %u (sum %08x) against %u (sum %08x)\n",
                  a.calls, a.call_sum, b.calls, b.call_sum);
    out += line;
  }
  if (a.coprocessor_words != b.coprocessor_words ||
      a.coprocessor_sum != b.coprocessor_sum) {
    std::snprintf(line, sizeof(line),
                  "  coprocessors: told %08x against %08x%s\n",
                  a.coprocessor_sum, b.coprocessor_sum,
                  a.coprocessor_words != b.coprocessor_words
                      ? ", and their registers differ"
                      : "");
    out += line;
  }
  if (a.written != b.written) {
    int shown = 0;
    for (const auto &[addr, value] : a.written) {
      const auto other = b.written.find(addr);
      if (other == b.written.end() || other->second != value) {
        std::snprintf(
            line, sizeof(line), "  memory %08x: %02x against %s\n", addr, value,
            other == b.written.end() ? "nothing written" : "another value");
        out += line;
        if (++shown == 6) {
          break;
        }
      }
    }
    if (a.written.size() != b.written.size()) {
      std::snprintf(line, sizeof(line),
                    "  bytes written by calls: %zu against %zu\n",
                    a.written.size(), b.written.size());
      out += line;
    }
  }
  if (a.data != b.data) {
    for (size_t i = 0; i < a.data.size() && i < b.data.size(); i++) {
      if (a.data[i] != b.data[i]) {
        std::snprintf(line, sizeof(line),
                      "  page memory %08x: %02x against %02x (first)\n",
                      DATA_BASE + static_cast<u32>(i), a.data[i], b.data[i]);
        out += line;
        break;
      }
    }
  }
  return out;
}

void Show(const std::vector<u32> &program, const Start &start) {
  std::printf("  start: cpsr %08x fpscr %08x\n  ", start.cpsr, start.fpscr);
  for (int i = 0; i < 16; i++) {
    std::printf("r%d=%08x%s", i, start.regs[i], i == 7 ? "\n  " : " ");
  }
  std::printf("\n");
  if (program.empty()) {
    // A game's code: the first instructions at the entry
    for (u32 pc = start.regs[15]; pc < start.regs[15] + 4 * 60; pc += 4) {
      const auto word = g_corpus.words.find(pc);
      if (word == g_corpus.words.end()) {
        break;
      }
      std::printf("  %08x  %08x  %s\n", pc, word->second,
                  A32::DisassembleArm(word->second).c_str());
    }
  }
  for (size_t i = 0; i < program.size(); i++) {
    std::printf("  %08x  %08x  %s\n", CODE_BASE + static_cast<u32>(i) * 4,
                program[i], A32::DisassembleArm(program[i]).c_str());
  }
}

} // namespace

namespace {

// A program that the recompiler never comes back from (it should, when the time
// given is up) is named instead of waited for
volatile unsigned long long g_program_running = 0;
volatile int g_side_running = 0;
unsigned g_seed_running = 0;
std::uint64_t g_slice = 0;

void OnTooLong(int) {
  char line[160];
  const int length = std::snprintf(
      line, sizeof(line),
      "program %llu (seed %u) DOES NOT COME BACK on the side of %s\n",
      g_program_running, g_seed_running,
      g_side_running == 0 ? "the reference" : "the test");
  if (length > 0) {
    (void)!write(1, line, static_cast<size_t>(length));
  }
  _exit(3);
}

/// The recompiler gave up on a program (an assertion of its own): which one
void OnAbort(int) {
  char line[160];
  const int length = std::snprintf(
      line, sizeof(line), "program %llu (seed %u) ABORTS on the side of %s\n",
      g_program_running, g_seed_running,
      g_side_running == 0 ? "the reference" : "the test");
  if (length > 0) {
    (void)!write(1, line, static_cast<size_t>(length));
  }
  _exit(4);
}

/// stuck SWITCHES: a program that goes round and round by returns to an address it has made up,
/// which is nowhere the recompiler has it look at its time: it has to come back all the same
int Stuck(u32 switches) {
  Machine m = MakeMachine(Mode::HostMmu, switches);
  // ldr lr, [pc]; bx lr; and the address of the first of them
  const std::vector<u32> program{0xE59FE000, 0xE12FFF1E, CODE_BASE};
  for (int i = 0; i < 3; i++) {
    m.env->Begin(program);
    m.jit->Regs()[15] = CODE_BASE;
    m.jit->SetCpsr(0x10);
    const u64 before = m.env->ticks_left;
    alarm(10);
    m.jit->Run();
    alarm(0);
    std::printf("came back with %" PRIu64 " of %" PRIu64
                " ticks left, at %08x\n",
                m.env->ticks_left, before, m.jit->Regs()[15]);
  }
  return 0;
}

} // namespace

/// demote SWITCHES: how long a copy loop takes, before and after its accesses have faulted as
/// often as it takes for them to be made the slow way
int Demote(u32 switches) {
  Machine m = MakeMachine(Mode::HostMmuFaults, switches);
  const std::vector<u32> program{
      0xE8B107F8, // ldm r1!, {r3-r10}
      0xE2522020, // subs r2, r2, #32
      0xE8A007F8, // stm r0!, {r3-r10}
      0xCAFFFFFB, // bgt to the first
      0xE5914000, // ldr r4, [r1]
      0xE12FFF1E, // bx lr
  };
  m.env->Begin(program);
  const auto copy = [&](u32 to, u32 from, u32 times) {
    const auto start = std::chrono::steady_clock::now();
    for (u32 i = 0; i < times; i++) {
      m.jit->Regs()[0] = to;
      m.jit->Regs()[1] = from;
      m.jit->Regs()[2] = 0x800;
      m.jit->Regs()[14] = 0x100;
      m.jit->Regs()[15] = CODE_BASE;
      m.jit->SetCpsr(0x10);
      m.env->stopped = false;
      m.env->ticks_left = 100000;
      while (!m.env->stopped && m.env->ticks_left > 0) {
        m.jit->Run();
      }
    }
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - start)
               .count() /
           (times * 64.0);
  };
  std::printf("a turn of the loop (32 bytes), ns: %.1f before\n",
              copy(DATA_BASE + 0x3000, DATA_BASE, 20000));
  // Page 1 takes no writes and page 2 nothing at all
  std::printf("  %.0f where it faults\n",
              copy(DATA_BASE + 0x1000, DATA_BASE, 3));
  std::printf("  %.1f after\n",
              copy(DATA_BASE + 0x3000, DATA_BASE, 20000));
  std::printf("  %.0f where it faulted\n",
              copy(DATA_BASE + 0x1000, DATA_BASE, 300));
  return 0;
}

/// modes SWITCHES: how long it takes to go to another floating point mode and back, and through a
/// loop in it
int Modes(u32 switches) {
  const auto time = [&](u32 bits, const char *name) {
    Machine m = MakeMachine(Mode::HostMmu, switches);
    const std::vector<u32> program{
        0xEEF11A10,        // vmrs r1, fpscr
        0xE3812000 | bits, // orr r2, r1, #bits
        0xE3A03014,        // 0x08: mov r3, #20
        0xEEE12A10,        // vmsr fpscr, r2
        0xE58D3000,        // 0x10: str r3, [sp]
        0xEE084A00,        // vmla.f32 s8, s16, s0
        0xE59D3000,        // ldr r3, [sp]
        0xEE4A4A20,        // vmla.f32 s9, s20, s1
        0xE2533001,        // subs r3, r3, #1
        0x1AFFFFF9,        // bne 0x10
        0xEEE11A10,        // vmsr fpscr, r1
        0xEB000003,        // bl 0x40
        0xE2500001,        // subs r0, r0, #1
        0x1AFFFFF3,        // bne 0x08
        0xE1A0F00B,        // mov pc, r11
        0xE1A00000,        // nop
        0xE12FFF1E,        // 0x40: bx lr
    };
    m.env->Begin(program);
    double ns = 0;
    for (int i = 0; i < 3; i++) {
      m.jit->SetCpsr(0x10);
      m.jit->SetFpscr(0x03000000);
      m.jit->Regs()[0] = 300;
      m.jit->Regs()[11] = 0x100;
      m.jit->Regs()[13] = DATA_BASE + 0x100;
      m.jit->Regs()[15] = CODE_BASE;
      m.env->stopped = false;
      m.env->ticks_left = 100000000;
      const auto start = std::chrono::steady_clock::now();
      while (!m.env->stopped && m.env->ticks_left > 0) {
        m.jit->Run();
      }
      ns = std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - start)
               .count() /
           300;
    }
    std::printf("%s: %.0f ns for there and back with 20 turns of the loop\n",
                name, ns);
  };
  time(0, "the same mode   ");
  time(0x501, "another rounding"); // #0x00400000
  time(0x803, "vectors of four "); // #0x00030000
  return 0;
}

/// calls SWITCHES: how long calls take of a routine that calls another one, from many places
int Calls(u32 switches) {
  Machine m = MakeMachine(Mode::HostMmu, switches);
  std::vector<u32> program{0xE3A04000}; // mov r4, #0
  constexpr u32 PLACES = 24;
  const u32 loop = static_cast<u32>(program.size());
  std::vector<u32> calls;
  for (u32 i = 0; i < PLACES; i++) {
    program.push_back(0xE2844001); // add r4, r4, #1
    calls.push_back(static_cast<u32>(program.size()));
    program.push_back(0);
  }
  program.push_back(0xE2500001); // subs r0, r0, #1
  program.push_back(0x1A000000 | ((loop - static_cast<u32>(program.size()) - 2) & 0xFFFFFF)); // bne
  program.push_back(0xE1A0F00B); // mov pc, r11
  const u32 routine = static_cast<u32>(program.size());
  program.push_back(0xE92D4020); // push {r5, lr}
  program.push_back(0xE0845004); // add r5, r4, r4
  const u32 inner = static_cast<u32>(program.size());
  program.push_back(0);
  program.push_back(0xE0844005); // add r4, r4, r5
  program.push_back(0xE8BD8020); // pop {r5, pc}
  const u32 leaf = static_cast<u32>(program.size());
  program.push_back(0xE0211004); // eor r1, r1, r4
  program.push_back(0xE12FFF1E); // bx lr
  for (const u32 call : calls) {
    // Every other one calls the inner routine
    program[call] = Branch((call % 4 == 1 ? leaf : routine) - call - 1, true, 0xE0000000);
  }
  program[inner] = Branch(leaf - inner - 1, true, 0xE0000000);
  m.env->Begin(program);
  for (int i = 0; i < 3; i++) {
    m.jit->SetCpsr(0x10);
    m.jit->Regs()[0] = 20000;
    m.jit->Regs()[11] = 0x100;
    m.jit->Regs()[13] = DATA_BASE + 0x800;
    m.jit->Regs()[15] = CODE_BASE;
    m.env->stopped = false;
    m.env->ticks_left = 100000000;
    const auto start = std::chrono::steady_clock::now();
    while (!m.env->stopped && m.env->ticks_left > 0) {
      m.jit->Run();
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    std::printf("%.2f ns a call from the loop\n", ns / (20000.0 * PLACES));
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "calls") == 0) {
    return Calls(static_cast<u32>(std::strtoul(argv[2], nullptr, 0)));
  }
  if (argc == 3 && std::strcmp(argv[1], "modes") == 0) {
    return Modes(static_cast<u32>(std::strtoul(argv[2], nullptr, 0)));
  }
  if (argc == 3 && std::strcmp(argv[1], "demote") == 0) {
    return Demote(static_cast<u32>(std::strtoul(argv[2], nullptr, 0)));
  }
  if (argc == 3 && std::strcmp(argv[1], "stuck") == 0) {
    signal(SIGALRM, OnTooLong);
    return Stuck(static_cast<u32>(std::strtoul(argv[2], nullptr, 0)));
  }
  if (argc < 6) {
    std::fprintf(stderr, "jitfuzz MODE(cb|pt|fm|ff) COUNT SEED REFERENCE TEST "
                         "[LENGTH] [simd] [spoil] [slice=N] [from=N] [show]\n");
    return 2;
  }
  const std::string mode_name = argv[1];
  const Mode mode = mode_name == "pt"   ? Mode::PageTable
                    : mode_name == "fm" ? Mode::HostMmu
                    : mode_name == "ff" ? Mode::HostMmuFaults
                                        : Mode::Callbacks;
  const u64 count = std::strtoull(argv[2], nullptr, 0);
  const u32 seed = static_cast<u32>(std::strtoul(argv[3], nullptr, 0));
  const u32 reference_switches =
      static_cast<u32>(std::strtoul(argv[4], nullptr, 0));
  const u32 test_switches = static_cast<u32>(std::strtoul(argv[5], nullptr, 0));
  const u32 length =
      argc > 6 ? static_cast<u32>(std::strtoul(argv[6], nullptr, 0)) : 24;
  bool spoil = false;
  u64 first = 0;
  bool show = false;
  u64 skip_first = 1, skip_last = 0;
  u64 cut = 0;
  for (int i = 7; i < argc; i++) {
    g_with_simd |= std::strcmp(argv[i], "simd") == 0;
    spoil |= std::strcmp(argv[i], "spoil") == 0;
    // from=N: begins at program N; show: prints every program it runs
    if (std::strncmp(argv[i], "from=", 5) == 0) {
      first = std::strtoull(argv[i] + 5, nullptr, 0);
    }
    show |= std::strcmp(argv[i], "show") == 0;
    // cut=K: only the first K instructions of each program are kept (what
    // follows them is where a program stops)
    if (std::strncmp(argv[i], "cut=", 4) == 0) {
      cut = std::strtoull(argv[i] + 4, nullptr, 0);
    }
    // skip=A-B: the programs from A to B are left out
    if (std::strncmp(argv[i], "skip=", 5) == 0) {
      char *rest = nullptr;
      skip_first = std::strtoull(argv[i] + 5, &rest, 0);
      skip_last = rest && *rest == '-' ? std::strtoull(rest + 1, nullptr, 0)
                                       : skip_first;
    }
    if (std::strcmp(argv[i], "modes=0") == 0) {
      g_with_modes = false;
    }
    if (std::strncmp(argv[i], "corpus=", 7) == 0) {
      LoadCorpus(argv[i] + 7);
    }
    if (std::strncmp(argv[i], "faults=", 7) == 0) {
      g_faults_before_recompile =
          static_cast<u32>(std::strtoul(argv[i] + 7, nullptr, 0));
    }
    // slice=N: the time is given in pieces of at most N
    if (std::strncmp(argv[i], "slice=", 6) == 0) {
      g_slice = std::strtoull(argv[i] + 6, nullptr, 0);
    }
  }
  g_seed_running = seed;
  signal(SIGALRM, OnTooLong);
  signal(SIGABRT, OnAbort);

  Machine reference = MakeMachine(mode, reference_switches);
  Machine test = MakeMachine(mode, test_switches);
  reference.env->slice = g_slice;
  test.env->slice = g_slice;

#if defined(__aarch64__)
  // A game's code: translated on a thread while the programs run
  std::atomic<bool> enough{false};
  std::thread ahead_thread;
  if (test.ahead && !g_corpus.words.empty()) {
    ahead_thread = std::thread{[&] {
      for (size_t i = seed % 7; i < g_corpus.entries.size() && !enough;
           i += 1 + i % 3) {
        const A32::LocationDescriptor location{
            g_corpus.entries[(i * 7919) % g_corpus.entries.size()],
            A32::PSR{0x10}, A32::FPSCR{0x03000000}};
        std::vector<u32> next{location.PC()};
        for (int depth = 0; depth < 3 && !enough; depth++) {
          std::vector<u32> after;
          for (const u32 pc : next) {
            const u64 at = location.SetPC(pc).UniqueHash();
            test.ahead->Translate(at, *test.ahead_memory);
            const auto leads = test.ahead->LeadsTo(at);
            after.insert(after.end(), leads.begin(), leads.end());
          }
          next = std::move(after);
        }
      }
    }};
  }
#endif

  u64 compared = 0, unsettled = 0, one_sided = 0, different = 0,
      kernel_calls = 0;
  for (u64 n = first; n < first + count; n++) {
    if (n >= skip_first && n <= skip_last) {
      continue;
    }
    // Each program from a seed of its own, so one can be run again alone
    detail::g_rand_int_generator.seed(seed * 1000003u + static_cast<u32>(n));
    std::vector<u32> program;
    if (g_corpus.words.empty()) {
      program = MakeProgram(4 + Bits(length));
    }
    const Start start =
        g_corpus.words.empty() ? MakeStart() : MakeCorpusStart();
    if (cut != 0 && program.size() > cut) {
      program.resize(cut);
    }
    if (show) {
      std::printf("program %" PRIu64 ":\n", n);
      Show(program, start);
      std::fflush(stdout);
    }
    g_program_running = n;
    g_side_running = 0;
    // (With the first one the decode tables are made)
    alarm(g_corpus.words.empty() && n != first ? 5 : 120);
    const Result a = RunOn(reference, program, start, 0);
    g_side_running = 1;
    const Result b =
        RunOn(test, program, start, spoil && n % 50 == 7 ? 0x00010000 : 0);
    alarm(0);
    if (!a.stopped || !b.stopped) {
      unsettled++;
      if (a.stopped != b.stopped) {
        one_sided++;
        if (one_sided <= 3) {
          std::printf("program %" PRIu64 " (seed %u): only %s came to an end\n",
                      n, seed, a.stopped ? "the reference" : "the test");
          Show(program, start);
        }
      }
      continue;
    }
    compared++;
    kernel_calls += a.calls;
    const std::string difference = Difference(a, b);
    if (!difference.empty()) {
      different++;
      if (different <= 5) {
        std::printf("program %" PRIu64
                    " (seed %u) DIFFERS, reference against test:\n%s",
                    n, seed, difference.c_str());
        Show(program, start);
      }
    }
  }
#if defined(__aarch64__)
  enough = true;
  if (ahead_thread.joinable()) {
    ahead_thread.join();
  }
  if (test.ahead) {
    const auto s = test.ahead->GetStatistics();
    std::printf("mode %s: translated ahead: %zu regions with %zu locations in "
                "%zu KB there; %zu along, %zu taken, %zu stale, %zu not there\n",
                mode_name.c_str(), s.regions, s.locations, s.host_bytes / 1024,
                s.along, s.taken, s.stale, s.missed);
  }
#endif
  std::printf("mode %s, switches %#x against %#x, seed %u: %" PRIu64
              " programs, %" PRIu64 " compared (%" PRIu64
              " kernel calls), %" PRIu64
              " still running when their time was up (%" PRIu64
              " of them on one side only), %" PRIu64 " DIFFERENT\n",
              mode_name.c_str(), reference_switches, test_switches, seed, count,
              compared, kernel_calls, unsettled, one_sided, different);
  if (Env::said_read_only != 0) {
    std::printf("mode %s: %" PRIu64 " loads were made constants\n",
                mode_name.c_str(), Env::said_read_only);
  }
  if (const u64 shared = Optimization::A32SharedHostAddressAccesses();
      shared != 0) {
    std::printf("mode %s: %" PRIu64
                " accesses were made at a shared host address\n",
                mode_name.c_str(), shared);
  }
#if defined(__aarch64__)
  if (const auto &direct = Backend::Arm64::Direct::statistics;
      direct.regions != 0) {
    std::printf("mode %s: translated directly %" PRIu64 " regions, %" PRIu64
                " blocks, %" PRIu64 " instructions into %" PRIu64
                " of the host's; %" PRIu64 " accesses carried out by calls\n",
                mode_name.c_str(), direct.regions.load(), direct.blocks.load(),
                direct.instructions.load(), direct.host_instructions.load(),
                direct.slow_accesses.load());
  }
#endif
  return different || one_sided ? 1 : 0;
}
