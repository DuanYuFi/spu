// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
namespace {
constexpr int64_t kWidth = 65536;
constexpr int64_t kDepth = 16;
constexpr size_t kBits = kWidth * kDepth * 64;
constexpr int kRepetitions = 7;

// Fully specified reproducible plaintext workload, not cryptographic
// randomness.
uint64_t InputWord(uint64_t index) {
  uint64_t z = index + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
class Barrier {
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_ = 0, generation_ = 0;

 public:
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto generation = generation_;
    if (++arrived_ == 3) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != generation; });
    }
  }
};
struct Result {
  double online_ms = 0, verify_ms = 0;
  size_t online_bytes = 0, verify_bytes = 0;
  NdArrayRef output;
};

void Run(bool native_aby3) {
  SpdzWiseIo io(FM64, 3);
  std::vector<std::vector<NdArrayRef>> shares;
  std::vector<uint64_t> expected(kWidth, ~uint64_t{0});
  for (int64_t layer = 0; layer <= kDepth; ++layer) {
    NdArrayRef input(makeType<RingTy>(FM64), {kWidth});
    for (int64_t i = 0; i < kWidth; ++i) {
      const auto word = InputWord(layer * kWidth + i);
      input.data<uint64_t>()[i] = word;
      expected[i] &= word;
    }
    shares.push_back(io.toBooleanShares(input, 64));
  }
  std::cout
      << std::fixed << std::setprecision(6)
      << "protocol,width,depth,word_ands,bit_ands,bucket_size,iteration,rank,"
         "online_ms,verify_ms,total_ms,online_bytes,verify_bytes,checks\n";
  for (int iteration = 0; iteration <= kRepetitions; ++iteration) {
    Barrier barrier;
    const auto results = utils::simulate(3, [&](const auto& link) {
      RuntimeConfig config;
      config.protocol = native_aby3 ? ABY3 : SPDZWISE;
      config.field = FM64;
      config.fxp_fraction_bits = 16;
      SPUContext ctx(config, link);
      if (native_aby3) {
        regAby3Protocol(&ctx, link);
      } else {
        registerArithmeticPrototype(&ctx, link);
      }
      auto* state =
          native_aby3 ? nullptr : ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      std::vector<Value> inputs;
      std::vector<std::string> names;
      for (int64_t layer = 0; layer <= kDepth; ++layer) {
        // The external XOR shares have the same replicated layout in both
        // protocols. Only benchmark type adaptation occurs outside timing.
        auto share = shares[layer][link->Rank()];
        if (native_aby3) share = share.as(makeType<aby3::BShrTy>(PT_U64, 64));
        inputs.emplace_back(share, DT_U64);
        names.push_back(std::to_string(layer));
      }
      if (state) {
        state->begin(ExecutableProto("abfl_c2", names, {"z"}, "benchmark"),
                     inputs);
        for (auto& input : inputs) input = import_s(&ctx, input);
      }
      Value z = inputs[0];
      Result result;
      const auto stats = link->GetStats();
      auto phase = [&](double& ms, size_t& bytes, auto&& operation) {
        barrier.wait();
        const auto before = stats->sent_bytes.load();
        const auto start = std::chrono::steady_clock::now();
        operation();
        ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start)
                 .count();
        bytes = stats->sent_bytes.load() - before;
      };
      phase(result.online_ms, result.online_bytes, [&] {
        for (int64_t layer = 0; layer < kDepth; ++layer) {
          z = dynDispatch(&ctx, "and_bb", z, inputs[layer + 1]);
        }
      });
      SPU_ENFORCE(result.online_bytes == kWidth * kDepth * sizeof(uint64_t));
      if (state) {
        SPU_ENFORCE(state->pendingBooleanBits() == kBits);
        SPU_ENFORCE(state->booleanChecks() == 0);
        phase(result.verify_ms, result.verify_bytes,
              [&] { state->finalize(); });
        SPU_ENFORCE(state->pendingBooleanBits() == 0 &&
                    state->booleanChecks() == 1);
        z.setDtype(DT_U64);
        const auto before_export = stats->sent_bytes.load();
        state->prepareOutputs(&ctx, {"z"}, {z});
        guard.commit();
        SPU_ENFORCE(stats->sent_bytes.load() == before_export);
        SPU_ENFORCE(state->booleanChecks() == 1);
        result.output = state->exportedValue("z").data();
      } else {
        result.output = z.data().as(makeType<BInputTy>(link->Rank(), 64));
        guard.commit();
      }
      return result;
    });
    std::vector<NdArrayRef> output;
    for (const auto& result : results) {
      output.push_back(result.output);
    }
    const auto actual = io.fromShares(output);
    for (int64_t i = 0; i < kWidth; ++i) {
      SPU_ENFORCE(actual.data<uint64_t>()[i] == expected[i],
                  "C2 output mismatch at {}", i);
    }
    double online_max = 0, verify_max = 0;
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& r = results[rank];
      online_max = std::max(online_max, r.online_ms);
      verify_max = std::max(verify_max, r.verify_ms);
      std::cout << (native_aby3 ? "aby3-boolean," : "spdzwise-abfl,") << kWidth
                << ',' << kDepth << ',' << kWidth * kDepth << ',' << kBits
                << ',' << (native_aby3 ? 0 : AbflBucketSize(kWidth * kDepth))
                << ',' << iteration << ',' << rank << ',' << r.online_ms << ','
                << r.verify_ms << ',' << r.online_ms + r.verify_ms << ','
                << r.online_bytes << ',' << r.verify_bytes << ","
                << (native_aby3 ? 0 : 1) << "\n";
    }
    std::cout.flush();
    std::cerr << "iteration " << iteration
              << (iteration == 0 ? " (warmup)" : "")
              << ": online_max_ms=" << online_max
              << ", verify_max_ms=" << verify_max << "; checked all " << kWidth
              << " outputs\n";
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise

int main(int argc, char** argv) {
  SPU_ENFORCE(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--aby3"),
              "usage: binary_bench [--aby3]");
  spu::mpc::spdzwise::Run(argc == 2);
}
