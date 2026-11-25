// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// directbench: how long the direct translator (A32, arm64) takes for a program's code.
//
// directbench CORPUS [PASSES]: the corpus as for jitfuzz, lines "w ADDRESS WORD" and "e ADDRESS",
// numbers in hexadecimal. Every entry is translated, as they come.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include <sys/mman.h>

#include "dynarmic/backend/arm64/a32_direct_translator.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/A32/coprocessor.h"
#include "dynarmic/interface/A32/translated_ahead.h"

namespace {

using namespace Dynarmic;

class Env final : public A32::UserCallbacks {
public:
  static constexpr u32 PAGE_BITS = 12;
  std::vector<std::unique_ptr<u32[]>> pages{1u << (32 - PAGE_BITS)};

  void Set(u32 address, u32 word) {
    auto &page = pages[address >> PAGE_BITS];
    if (!page) {
      page = std::make_unique<u32[]>(1u << (PAGE_BITS - 2));
    }
    page[(address >> 2) & ((1u << (PAGE_BITS - 2)) - 1)] = word;
  }
  std::optional<std::uint32_t> MemoryReadCode(u32 vaddr) override {
    const auto &page = pages[vaddr >> PAGE_BITS];
    if (!page) {
      return std::nullopt;
    }
    const u32 word = page[(vaddr >> 2) & ((1u << (PAGE_BITS - 2)) - 1)];
    return word ? std::optional<std::uint32_t>{word} : std::nullopt;
  }
  bool IsReadOnlyMemory(u32 vaddr) override {
    return vaddr >= 0x00100000 && vaddr < 0x04000000;
  }
  std::uint8_t MemoryRead8(u32) override { return 0; }
  std::uint16_t MemoryRead16(u32) override { return 0; }
  std::uint32_t MemoryRead32(u32 vaddr) override {
    return MemoryReadCode(vaddr & ~3u).value_or(0x12345678);
  }
  std::uint64_t MemoryRead64(u32) override { return 0; }
  void MemoryWrite8(u32, std::uint8_t) override {}
  void MemoryWrite16(u32, std::uint16_t) override {}
  void MemoryWrite32(u32, std::uint32_t) override {}
  void MemoryWrite64(u32, std::uint64_t) override {}
  void InterpreterFallback(u32, size_t) override {}
  void CallSVC(std::uint32_t) override {}
  void ExceptionRaised(u32, A32::Exception) override {}
  void AddTicks(std::uint64_t) override {}
  std::uint64_t GetTicksRemaining() override { return 0; }
};

} // namespace

/// Words, whatever is asked for
class Words final : public A32::Coprocessor {
public:
  using CoprocReg = A32::CoprocReg;
  u32 word = 0;
  std::optional<Callback> CompileInternalOperation(bool, unsigned, CoprocReg, CoprocReg, CoprocReg,
                                                   unsigned) override {
    return Callback{[](void *, u32, u32) -> u64 { return 0; }, std::nullopt};
  }
  CallbackOrAccessOneWord CompileSendOneWord(bool, unsigned, CoprocReg, CoprocReg,
                                             unsigned) override {
    return &word;
  }
  CallbackOrAccessTwoWords CompileSendTwoWords(bool, unsigned, CoprocReg) override {
    return std::array<u32 *, 2>{&word, &word};
  }
  CallbackOrAccessOneWord CompileGetOneWord(bool, unsigned, CoprocReg, CoprocReg,
                                            unsigned) override {
    return &word;
  }
  CallbackOrAccessTwoWords CompileGetTwoWords(bool, unsigned, CoprocReg) override {
    return std::array<u32 *, 2>{&word, &word};
  }
  std::optional<Callback> CompileLoadWords(bool, bool, CoprocReg,
                                           std::optional<std::uint8_t>) override {
    return std::nullopt;
  }
  std::optional<Callback> CompileStoreWords(bool, bool, CoprocReg,
                                            std::optional<std::uint8_t>) override {
    return std::nullopt;
  }
};

class AheadMemory final : public A32::TranslatedAhead::Memory {
public:
  Env &env;
  explicit AheadMemory(Env &env) : env{env} {}
  std::optional<std::uint32_t> Read(std::uint32_t vaddr) override {
    return env.MemoryReadCode(vaddr);
  }
  bool IsReadOnly(std::uint32_t vaddr) override {
    return env.IsReadOnlyMemory(vaddr);
  }
  std::uint64_t GetTicksForCode(std::uint32_t, std::uint32_t) override {
    return 1;
  }
};

/// With a recompiler: the code of every entry put in place, translated ahead or not. It is never
/// run: the recompiler is told to stop before it starts.
void Place(Env &env, const std::vector<u32> &entries, u8 *view, bool ahead_wanted) {
  A32::TranslatedAhead ahead;
  A32::UserConfig config;
  config.callbacks = &env;
  config.fastmem_pointer = reinterpret_cast<std::uintptr_t>(view);
  config.fastmem_guard_size = u64{8} << 30;
  config.define_unpredictable_behaviour = true;
  config.pin_guest_registers = true;
  config.write_into_guest_registers = true;
  config.fold_conditional_branches = true;
  config.guest_flags_in_host_flags = true;
  config.fp_status_in_host = true;
  config.fp_registers_through_general = true;
  config.look_up_blocks_in_place = true;
  config.pin_guest_fp_registers = true;
  config.conditional_instructions_in_blocks = true;
  config.fold_read_only_loads = true;
  config.translate_directly = true;
  config.translated_ahead = &ahead;
  config.coprocessors[15] = std::make_shared<Words>();
  A32::Jit jit{config};
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto ms = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  const auto start = now();
  if (ahead_wanted) {
    AheadMemory memory{env};
    for (const u32 entry : entries) {
      const A32::LocationDescriptor location{entry, A32::PSR{0x10},
                                             A32::FPSCR{0x03000000}};
      ahead.Translate(location.UniqueHash(), memory);
    }
  }
  const auto between = now();
  jit.SetCpsr(0x10);
  jit.SetFpscr(0x03000000);
  double longest = 0;
  for (const u32 entry : entries) {
    const auto from = now();
    jit.HaltExecution();
    jit.Regs()[15] = entry;
    jit.Run();
    jit.ClearHalt();
    longest = std::max(longest, ms(now() - from));
  }
  const auto end = now();
  const auto s = ahead.GetStatistics();
  std::printf("%s: ahead %.1f ms (%zu regions, %zu locations, %zu KB); then %.1f ms: "
              "%zu taken and %zu along, %zu translated in %.1f ms, %.1f ms to put in "
              "place, %zu refused, longest %.2f ms\n",
              ahead_wanted ? "translated ahead" : "when needed", ms(between - start),
              s.regions, s.locations, s.host_bytes / 1024, ms(end - between), s.taken,
              s.along, s.missed, static_cast<double>(s.translating_ns) / 1e6,
              static_cast<double>(s.placing_ns) / 1e6, s.refused, longest);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    return 2;
  }
  Env env;
  std::vector<u32> entries;
  std::FILE *const file = std::fopen(argv[1], "r");
  if (!file) {
    std::perror(argv[1]);
    return 2;
  }
  char line[128];
  size_t words = 0;
  while (std::fgets(line, sizeof(line), file)) {
    unsigned a = 0, b = 0;
    if (std::sscanf(line, "w %x %x", &a, &b) == 2) {
      env.Set(a, b);
      words++;
    } else if (std::sscanf(line, "e %x", &a) == 1) {
      entries.push_back(a);
    }
  }
  std::fclose(file);
  const int passes = argc > 2 ? std::atoi(argv[2]) : 3;

  if (argc > 3) {
    const size_t size = (u64{4} << 30) * 2 + (u64{8} << 30) * 2;
    u8 *const reserved = static_cast<u8 *>(
        mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (reserved == MAP_FAILED) {
      std::perror("mmap");
      return 2;
    }
    u8 *const view = reinterpret_cast<u8 *>(
        (reinterpret_cast<u64>(reserved) + (u64{8} << 30) + 0xFFFFFFFF) & ~u64{0xFFFFFFFF});
    for (int pass = 0; pass < passes; pass++) {
      Place(env, entries, view, false);
      Place(env, entries, view, true);
    }
    return 0;
  }

  for (int pass = 0; pass < passes; pass++) {
    std::unordered_set<u64> known;
    known.reserve(entries.size() * 8);
    Backend::Arm64::Direct::TranslatorConfig config;
    config.callbacks = &env;
    config.fold_read_only_loads = true;
    config.guard_size = u64{8} << 30;
    config.pinned_fp_registers = 24;
    config.entry_table = true;
    config.unlooked_returns = true;
    config.fixed_bases = true;
    config.state_pc_offset = 0x54;
    config.state_upper_offset = 0x58;
    config.state_ext_regs_offset = 0x60;
    config.state_fpsr_nzcv_offset = 0x14;
    config.stack_dispatch_table_offset = 0x498;
    config.is_known = [&](IR::LocationDescriptor d) {
      return known.count(d.Value()) != 0;
    };
    config.is_checked = [](u32) { return false; };

    Backend::Arm64::Direct::Translator translator{config};
    size_t regions = 0, instructions = 0, host = 0, refused = 0;
    long longest = 0, over_500 = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const u32 entry : entries) {
      const A32::LocationDescriptor location{entry, A32::PSR{0x10},
                                             A32::FPSCR{0x03000000}};
      if (known.count(location.UniqueHash())) {
        continue;
      }
      const auto from = std::chrono::steady_clock::now();
      const auto region = translator.Translate(location);
      const long took =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - from)
              .count();
      longest = std::max(longest, took);
      over_500 += took > 500;
      if (!region) {
        refused++;
        continue;
      }
      for (const auto &[descriptor, offset] : region->entries) {
        known.insert(descriptor.Value());
      }
      regions++;
      instructions += region->num_instructions;
      host += region->code.size();
    }
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    std::printf("%zu words, %zu entries: %zu regions (%zu refused), %zu "
                "instructions, %zu KB of code, %zu blocks; %.1f ms, %.1f us a "
                "region, %.3f us an instruction, longest %ld us, %ld over 500 us\n",
                words, entries.size(), regions, refused, instructions,
                host * 4 / 1024, known.size(), us / 1000, us / regions,
                us / instructions, longest, over_500);
  }
  return 0;
}
