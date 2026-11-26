// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// directdump: what the direct translator (A32, arm64) makes of a program's code.
//
// Standard input: lines "w ADDRESS WORD" (guest memory), "f FPSCR" (its mode), "n COUNT" (the
// most instructions at once) and "e ADDRESS" (translate from there, unless it has been already),
// numbers in hexadecimal. Output for each region: "r ENTRY INSTRUCTIONS HOST_WORDS", then for
// each piece of it "p GUEST_PC OFFSET" and its code as "c WORD...".

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>

#include "dynarmic/backend/arm64/a32_direct_translator.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/interface/A32/config.h"

namespace {

using namespace Dynarmic;

class Env final : public A32::UserCallbacks {
public:
  std::map<u32, u32> words;

  std::optional<std::uint32_t> MemoryReadCode(u32 vaddr) override {
    const auto found = words.find(vaddr);
    return found == words.end() ? std::nullopt
                                : std::optional<std::uint32_t>(found->second);
  }
  bool IsReadOnlyMemory(u32 vaddr) override {
    return vaddr >= 0x00100000 && vaddr < 0x04000000;
  }
  std::uint8_t MemoryRead8(u32) override { return 0; }
  std::uint16_t MemoryRead16(u32) override { return 0; }
  std::uint32_t MemoryRead32(u32 vaddr) override {
    const auto found = words.find(vaddr);
    return found == words.end() ? 0x12345678 : found->second;
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

int main() {
  Env env;
  std::set<u64> known;
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
  u32 fpscr = 0;

  char line[256];
  while (std::fgets(line, sizeof(line), stdin)) {
    unsigned a = 0, b = 0;
    if (std::sscanf(line, "w %x %x", &a, &b) == 2) {
      env.words[a] = b;
    } else if (std::sscanf(line, "f %x", &a) == 1) {
      fpscr = a;
    } else if (std::sscanf(line, "n %x", &a) == 1) {
      config.most_instructions = a;
    } else if (std::sscanf(line, "s %x", &a) == 1) {
      config.pinned_fp_registers = a;
    } else if (std::sscanf(line, "C %x", &a) == 1) {
      config.literal_constants = a != 0;
    } else if (std::sscanf(line, "A %x", &a) == 1) {
      config.literal_addresses = a != 0;
    } else if (std::sscanf(line, "H %x", &a) == 1) {
      config.host_returns = a != 0;
    } else if (std::sscanf(line, "e %x", &a) == 1) {
      const A32::LocationDescriptor location{a, A32::PSR{0x10},
                                             A32::FPSCR{fpscr}};
      if (known.count(location.UniqueHash())) {
        continue;
      }
      Backend::Arm64::Direct::Translator translator{config};
      const auto region = translator.Translate(location);
      if (!region) {
        std::printf("x %x\n", a);
        continue;
      }
      for (const auto &[descriptor, offset] : region->entries) {
        known.insert(descriptor.Value());
      }
      std::printf("r %x %zx %zx\n", a, region->num_instructions,
                  region->code.size());
      for (const auto &[offset, pc] : region->instruction_offsets) {
        std::printf("p %x %x\n", pc, offset);
      }
      std::printf("c");
      for (const u32 word : region->code) {
        std::printf(" %x", word);
      }
      std::printf("\n");
    }
  }
  return 0;
}
