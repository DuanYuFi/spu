// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>

#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/field.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
namespace {
class Barrier {
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_ = 0;
  size_t generation_ = 0;

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
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
struct Result {
  // Normal input import (including its immediate check), local expansion,
  // circuit (add + multiply + saved check records), final MAC check.
  std::array<double, 4> ms{};
  std::array<size_t, 4> bytes{};
  double mul_ms = 0;
  NdArrayRef output;
};

void Run(int64_t width, int64_t depth, int repetitions, bool ring = true) {
  const auto gates = width * depth;
  RuntimeConfig config(SPDZWISE, FM64);
  config.spdzwise_config.domain =
      ring ? SpdzWiseConfig::RING : SpdzWiseConfig::FIELD;
  config.fxp_fraction_bits = 16;
  SpdzWiseIo io(FM64, 3, config.spdzwise_config.domain);
  NdArrayRef plain(makeType<RingTy>(FM64), {2});
  // Same raw input words and recurrence as the existing arithmetic benchmark.
  plain.data<uint64_t>()[0] = (uint64_t{1} << 60) - 8;
  plain.data<uint64_t>()[1] = uint64_t(-13);
  const auto inputs = io.toShares(plain, VIS_SECRET);
  NdArrayRef offsets(makeType<Pub2kTy>(FM64), {width});
  std::vector<uint64_t> expected(width);
  for (int64_t i = 0; i < width; ++i) {
    offsets.data<uint64_t>()[i] = i;
    auto x = ring ? plain.data<uint64_t>()[0] + uint64_t(i)
                  : Add(FromRing(plain.data<uint64_t>()[0]), i);
    for (int64_t layer = 0; layer < depth; ++layer) {
      x = ring ? x * (x + plain.data<uint64_t>()[1])
               : Mul(x, Add(x, FromRing(plain.data<uint64_t>()[1])));
    }
    expected[i] = ring ? x : ToRing(x);
  }
  std::cout << std::fixed << std::setprecision(6)
            << "protocol,width,depth,gates,secret_inputs,iteration,rank,"
               "input_ms,expand_ms,circuit_ms,verify_ms,mul_ms,"
               "input_bytes,expand_bytes,circuit_bytes,verify_bytes,checks\n";
  // A new session each iteration; setup and checked export are outside timing.
  // Iteration zero is the warmup, retained in CSV but excluded from statistics.
  for (int iteration = 0; iteration <= repetitions; ++iteration) {
    Barrier barrier;
    const auto results = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(config, link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Value original(inputs[link->Rank()], DT_I64);
      state->begin(ExecutableProto("ring_circuit_bench", {"xy"}, {"z"}, "test"),
                   {original});
      const auto checks = state->arithmeticChecks();
      const auto stats = link->GetStats();
      Result result;
      auto phase = [&](size_t index, auto&& fn) {
        barrier.wait();
        const auto bytes = stats->sent_bytes.load();
        const auto start = Clock::now();
        fn();
        result.ms[index] = Milliseconds(start);
        result.bytes[index] = stats->sent_bytes.load() - bytes;
      };
      Value authenticated, z, y;
      phase(0, [&] { authenticated = import_s(&ctx, original); });
      SPU_ENFORCE(state->arithmeticChecks() == checks + 1);
      phase(1, [&] {
        z = dynDispatch(&ctx, "extract_slice", authenticated, Index{0},
                        Index{1}, Strides{1});
        y = dynDispatch(&ctx, "extract_slice", authenticated, Index{1},
                        Index{2}, Strides{1});
        z = dynDispatch(&ctx, "broadcast", z, Shape{width}, Axes{0});
        y = dynDispatch(&ctx, "broadcast", y, Shape{width}, Axes{0});
        z = dynDispatch(&ctx, "add_ap", z, Value(offsets, DT_I64));
      });
      phase(2, [&] {
        for (int64_t layer = 0; layer < depth; ++layer) {
          const auto rhs = dynDispatch(&ctx, "add_aa", z, y);
          const auto start = Clock::now();
          z = dynDispatch(&ctx, "mul_aa", z, rhs);
          result.mul_ms += Milliseconds(start);
        }
      });
      SPU_ENFORCE(state->arithmeticChecks() == checks + 1,
                  "unexpected check within the circuit");
      phase(3, [&] { state->check(); });
      SPU_ENFORCE(state->arithmeticChecks() == checks + 2);
      // Per-party payloads: each resharing sends two 128-bit words per gate,
      // plus two all-gather status rounds (4 bytes) per layer. Verification
      // includes batch size, record copies, secret aggregation and Protocol 9.
      SPU_ENFORCE(result.bytes[0] == (ring ? 242 : 96) && result.bytes[1] == 0);
      SPU_ENFORCE(result.bytes[2] == size_t(gates) * (ring ? 32 : 16) +
                                         (ring ? size_t(depth) * 4 : 0));
      SPU_ENFORCE(result.bytes[3] ==
                  (ring ? 168 + 2 * std::to_string(gates).size() : 64));
      z.setDtype(DT_I64);
      state->prepareOutputs(&ctx, {"z"}, {z});
      guard.commit();
      SPU_ENFORCE(state->arithmeticChecks() == checks + 2,
                  "unexpected nonempty check at export");
      result.output = state->exportedValue("z").data();
      return result;
    });
    std::vector<NdArrayRef> outputs;
    for (const auto& result : results) outputs.push_back(result.output);
    const auto reconstructed = io.fromShares(outputs);
    const NdArrayView<uint64_t> actual(reconstructed);
    for (int64_t i = 0; i < width; ++i) {
      SPU_ENFORCE(actual[i] == expected[i],
                  "wrong output at iteration {}, index {}", iteration, i);
    }
    std::cerr << "iteration " << iteration << ": checked all " << width
              << " outputs; " << gates << " multiplication gates\n";
    for (size_t rank = 0; rank < results.size(); ++rank) {
      const auto& result = results[rank];
      std::cout << (ring ? "spdzwise-ring128," : "spdzwise-f61-normal,")
                << width << ',' << depth << ',' << gates << ",2," << iteration
                << ',' << rank;
      for (const auto ms : result.ms) std::cout << ',' << ms;
      std::cout << ',' << result.mul_ms;
      for (const auto bytes : result.bytes) std::cout << ',' << bytes;
      std::cout << ",2\n";
    }
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise

int main(int argc, char** argv) {
  SPU_ENFORCE(argc == 4 || (argc == 5 && std::string(argv[4]) == "f61"),
              "usage: ring_circuit_bench width depth repetitions [f61]");
  const int64_t width = std::stoll(argv[1]), depth = std::stoll(argv[2]);
  const int repetitions = std::stoi(argv[3]);
  SPU_ENFORCE(width > 0 && depth > 0 && repetitions > 0 &&
              width <= std::numeric_limits<int64_t>::max() / depth);
  spu::mpc::spdzwise::Run(width, depth, repetitions, argc == 4);
}
