// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "libspu/kernel/hal/integer.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/conversion.h"
#include "libspu/mpc/aby3/io.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/ring_conversion.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
namespace {
using Clock = std::chrono::steady_clock;
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
uint64_t SplitMix(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
struct Result {
  double total_ms = 0, circuit_ms = 0, finalize_output_ms = 0;
  size_t sent_bytes = 0, sent_actions = 0, and_bits = 0;
  size_t circuit_bytes = 0, finalize_output_bytes = 0;
  size_t check_mac_calls = 0, check_boolean_calls = 0;
  size_t mac_batches = 0, boolean_batches = 0;
  NdArrayRef output;
};
void Run(const std::string& protocol, int64_t count, int repetitions) {
  const bool wise = protocol == "spdzwise";
  NdArrayRef px(makeType<RingTy>(FM64), {count}), py(px.eltype(), px.shape());
  const std::array<int64_t, 5> edges{0, -1, 1, -(int64_t{1} << 62),
                                    (int64_t{1} << 62) - 1};
  uint64_t seed = 0x4d5342434d503634ULL;
  for (int64_t i = 0; i < count; ++i) {
    px.data<uint64_t>()[i] = i < 25 ? edges[i / 5]
        : int64_t(SplitMix(seed) >> 1) - (int64_t{1} << 62);
    py.data<uint64_t>()[i] = i < 25 ? edges[i % 5]
        : int64_t(SplitMix(seed) >> 1) - (int64_t{1} << 62);
  }
  SpdzWiseIo wise_io(FM64, 3, SpdzWiseConfig::RING);
  auto aby_io = aby3::makeAby3Io(FM64, 3);
  RuntimeConfig config(wise ? SPDZWISE : ABY3, FM64);
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  std::cout << std::fixed << std::setprecision(6)
            << "protocol,count,iteration,rank,total_ms,circuit_ms,finalize_output_ms,"
               "sent_bytes,sent_actions,circuit_bytes,finalize_output_bytes,and_bits,"
               "check_mac_calls,check_boolean_calls,mac_batches,boolean_batches,"
               "outputs_verified\n";
  for (int iteration = 0; iteration <= repetitions; ++iteration) {
    const auto sx = wise ? wise_io.toShares(px, VIS_SECRET)
                         : aby_io->toShares(px, VIS_SECRET, -1);
    const auto sy = wise ? wise_io.toShares(py, VIS_SECRET)
                         : aby_io->toShares(py, VIS_SECRET, -1);
    Barrier barrier;
    const auto results = utils::simulate(3, [&](const auto& link) {
      std::unique_ptr<SPUContext> ctx =
          wise ? std::make_unique<SPUContext>(config, link)
               : makeAby3Protocol(config, link);
      if (wise) registerArithmeticPrototype(ctx.get(), link);
      auto* state = wise ? ctx->prot()->getState<SpdzWiseState>() : nullptr;
      ProtocolRunGuard guard(ctx.get());
      Value x(sx[link->Rank()], DT_I64), y(sy[link->Rank()], DT_I64);
      if (wise) {
        SPU_ENFORCE(dynamic_cast<ring::MsbA2B*>(ctx->prot()->getKernel("msb_a2b")));
        state->begin(ExecutableProto("comparison-bench", {"x", "y"},
                                    {"out"}, "test"), {x, y});
        x = import_s(ctx.get(), x);
        y = import_s(ctx.get(), y);
        SPU_ENFORCE(state->pendingBooleanBits() == 0);
      } else {
        SPU_ENFORCE(dynamic_cast<aby3::MsbA2B*>(ctx->prot()->getKernel("msb_a2b")));
      }
      const auto stats = link->GetStats();
      const auto arithmetic_checks = wise ? state->arithmeticChecks() : 0;
      const auto boolean_checks = wise ? state->booleanChecks() : 0;
      SPU_ENFORCE(x.shape() == Shape({count}) && y.shape() == x.shape() &&
                  x.dtype() == DT_I64 && y.dtype() == DT_I64 &&
                  x.storage_type().isa<AShare>() && y.storage_type().isa<AShare>());
      Result r;
      barrier.wait();
      const auto bytes = stats->sent_bytes.load();
      const auto actions = stats->sent_actions.load();
      const auto start = Clock::now();
      // Identical HAL operation: two signed FM64 arithmetic arrays -> one-bit
      // Boolean comparison array. Each protocol uses its registered MSB kernel.
      auto out = kernel::hal::i_less(ctx.get(), x, y);
      const auto circuit_stop = Clock::now();
      const auto circuit_bytes = stats->sent_bytes.load();
      if (wise) {
        r.and_bits = state->pendingBooleanBits();
        // The normal output transaction invokes finalize_protocol -> verify()
        // -> checkMac() + checkBoolean(), then exports checked Boolean shares.
        // Keep this entire operation inside the same timed boundary as ABY3.
        state->prepareOutputs(ctx.get(), {"out"}, {out});
        guard.commit();
        r.output = state->exportedValue("out").data();
      } else {
        r.output = out.data();
        guard.commit();
      }
      const auto stop = Clock::now();
      const auto elapsed = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
      };
      r.total_ms = elapsed(start, stop);
      r.circuit_ms = elapsed(start, circuit_stop);
      r.finalize_output_ms = elapsed(circuit_stop, stop);
      r.sent_bytes = stats->sent_bytes.load() - bytes;
      r.sent_actions = stats->sent_actions.load() - actions;
      r.circuit_bytes = circuit_bytes - bytes;
      r.finalize_output_bytes = r.sent_bytes - r.circuit_bytes;
      SPU_ENFORCE(out.shape() == x.shape() && out.dtype() == DT_I1 &&
                  out.storage_type().as<BShare>()->nbits() == 1 &&
                  r.output.shape() == x.shape() &&
                  r.output.eltype().as<BShare>()->nbits() == 1 &&
                  r.output.eltype().isSerializable());
      if (wise) {
        r.check_mac_calls = r.check_boolean_calls = 1;
        r.mac_batches = state->arithmeticChecks() - arithmetic_checks;
        r.boolean_batches = state->booleanChecks() - boolean_checks;
        // Already authenticated inputs plus linear subtraction leave no new
        // arithmetic batch. The empty checkMac count/status exchange is timed.
        SPU_ENFORCE(r.mac_batches == 0 && r.boolean_batches == 1 &&
                    state->pendingBooleanBits() == 0);
        SPU_ENFORCE(r.and_bits == 241 * ((size_t(count) + 63) / 64) * 64);
      }
      // Only reconstruction and the plaintext oracle remain outside timing.
      barrier.wait();
      return r;
    });
    std::vector<NdArrayRef> output;
    for (const auto& r : results) output.push_back(r.output);
    const auto actual = wise ? wise_io.fromShares(output) : aby_io->fromShares(output);
    // ABY3's one-bit shares may use a narrower backing type. Reconstruction
    // returns FM64 plaintext for both; never reinterpret raw ABY3 shares.
    for (int64_t i = 0; i < count; ++i) {
      const auto x = px.data<uint64_t>()[i], y = py.data<uint64_t>()[i];
      SPU_ENFORCE(actual.data<uint64_t>()[i] == ((x - y) >> 63) &&
                  actual.data<uint64_t>()[i] == uint64_t(int64_t(x) < int64_t(y)),
                  "comparison mismatch: iteration {}, element {}", iteration, i);
    }
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& r = results[rank];
      std::cout << protocol << ',' << count << ',' << iteration << ',' << rank
                << ',' << r.total_ms << ',' << r.circuit_ms << ','
                << r.finalize_output_ms << ',' << r.sent_bytes << ','
                << r.sent_actions << ',' << r.circuit_bytes << ','
                << r.finalize_output_bytes << ',' << r.and_bits << ','
                << r.check_mac_calls << ',' << r.check_boolean_calls << ','
                << r.mac_batches << ',' << r.boolean_batches << ',' << count << '\n';
    }
    std::cout.flush();
    std::cerr << protocol << " iteration " << iteration << ": verified all "
              << count << " comparisons outside timing\n";
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise

int main(int argc, char** argv) {
  try {
    SPU_ENFORCE(argc <= 4);
    const std::string protocol = argc > 1 ? argv[1] : "spdzwise";
    const auto count = argc > 2 ? std::stoll(argv[2]) : 100000;
    const auto repetitions = argc > 3 ? std::stoi(argv[3]) : 7;
    SPU_ENFORCE((protocol == "spdzwise" || protocol == "aby3") && count > 0 &&
                repetitions > 0);
    spu::mpc::spdzwise::Run(protocol, count, repetitions);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
