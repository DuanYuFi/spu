// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>

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
double Ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
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

uint64_t SplitMix(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
NdArrayRef Inputs(int64_t count) {
  NdArrayRef out(makeType<RingTy>(FM64), {count});
  const std::array<uint64_t, 8> edges{0, 1, ~uint64_t(0), uint64_t(1) << 63,
      (uint64_t(1) << 63) - 1, 0xaaaaaaaaaaaaaaaaULL, 0x5555555555555555ULL, 0xffffffffULL};
  uint64_t seed = 0x53504c4954334132ULL;
  for (int64_t i = 0; i < count; ++i)
    out.data<uint64_t>()[i] = size_t(i) < edges.size() ? edges[i] : SplitMix(seed);
  return out;
}
struct Result {
  double total_ms = 0, online_ms = 0, verify_ms = 0;
  size_t bytes = 0, actions = 0, online_bytes = 0, verify_bytes = 0,
         input_checks = 0, final_checks = 0, and_bits = 0;
  NdArrayRef output;
};
void Run(const std::string& protocol, int64_t count, int repetitions) {
  const bool wise = protocol != "aby3";
  const auto plain = Inputs(count);
  SpdzWiseIo wise_io(FM64, 3, SpdzWiseConfig::RING);
  auto aby_io = aby3::makeAby3Io(FM64, 3);
  RuntimeConfig config(wise ? SPDZWISE : ABY3, FM64);
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  std::cout << std::fixed << std::setprecision(6)
      << "protocol,count,iteration,rank,total_ms,online_ms,verify_ms,sent_bytes,online_bytes,verify_bytes,sent_actions,input_checks,final_checks,and_bits,outputs_verified\n";
  for (int iteration = 0; iteration <= repetitions; ++iteration) {
    const auto inputs = wise ? wise_io.toShares(plain, VIS_SECRET)
                            : aby_io->toShares(plain, VIS_SECRET, -1);
    Barrier barrier;
    const auto results = utils::simulate(3, [&](const auto& link) {
      std::unique_ptr<SPUContext> ctx = wise ? std::make_unique<SPUContext>(config, link)
                      : makeAby3Protocol(config, link);
      if (wise) registerArithmeticPrototype(ctx.get(), link);
      auto* state = wise ? ctx->prot()->getState<SpdzWiseState>() : nullptr;
      ProtocolRunGuard guard(ctx.get());
      Value x(inputs[link->Rank()], DT_U64);
      Result r;
      if (wise) {
        SPU_ENFORCE(dynamic_cast<ring::A2B*>(ctx->prot()->getKernel("a2b")));
        state->begin(ExecutableProto("a2b-bench", {"x"}, {"y"}, "test"), {x});
        x = import_s(ctx.get(), x);
        r.input_checks = state->arithmeticChecks();
        SPU_ENFORCE(r.input_checks == 1);
      } else {
        SPU_ENFORCE(dynamic_cast<aby3::A2B*>(ctx->prot()->getKernel("a2b")));
      }
      const auto stats = link->GetStats();
      barrier.wait();
      const auto bytes = stats->sent_bytes.load();
      const auto actions = stats->sent_actions.load();
      const auto start = Clock::now();
      auto y = a2b(ctx.get(), x);
      const auto online_stop = Clock::now();
      const auto online_bytes = stats->sent_bytes.load();
      if (wise) {
        r.and_bits = state->pendingBooleanBits();
        state->finalize();
      }
      const auto stop = Clock::now();
      r.total_ms = Ms(start, stop);
      r.online_ms = Ms(start, online_stop);
      r.verify_ms = wise ? Ms(online_stop, stop) : 0;
      r.bytes = stats->sent_bytes.load() - bytes;
      r.online_bytes = online_bytes - bytes;
      r.verify_bytes = r.bytes - r.online_bytes;
      r.actions = stats->sent_actions.load() - actions;
      if (!wise) y.setDtype(x.dtype());  // Native low-level kernels erase dtype.
      SPU_ENFORCE(y.shape() == x.shape() && y.dtype() == x.dtype() &&
                  y.storage_type().as<BShare>()->nbits() == 64);
      if (wise) {
        r.final_checks = state->booleanChecks();
        SPU_ENFORCE(r.final_checks == 1 && state->pendingBooleanBits() == 0);
        SPU_ENFORCE(state->arithmeticChecks() == r.input_checks);
        state->prepareOutputs(ctx.get(), {"y"}, {y});
        guard.commit();
        SPU_ENFORCE(state->booleanChecks() == 1 && state->arithmeticChecks() == r.input_checks);
        r.output = state->exportedValue("y").data();
      } else {
        r.output = y.data();
        guard.commit();
      }
      return r;
    });
    std::vector<NdArrayRef> output;
    for (const auto& r : results) output.push_back(r.output);
    const auto actual = wise ? wise_io.fromShares(output) : aby_io->fromShares(output);
    for (int64_t i = 0; i < count; ++i) {
      SPU_ENFORCE(actual.data<uint64_t>()[i] == plain.data<uint64_t>()[i],
                  "A2B output mismatch: iteration {}, element {}", iteration, i);
      for (size_t rank = 0; rank < 3; ++rank)
        SPU_ENFORCE(output[rank].data<RepShare>()[i][1] == output[(rank + 1) % 3].data<RepShare>()[i][0]);
    }
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& r = results[rank];
      std::cout << protocol << ',' << count << ',' << iteration << ',' << rank << ','
          << r.total_ms << ',' << r.online_ms << ',' << r.verify_ms << ',' << r.bytes << ','
          << r.online_bytes << ',' << r.verify_bytes << ',' << r.actions << ','
          << r.input_checks << ',' << r.final_checks << ',' << r.and_bits << ',' << count << '\n';
    }
    std::cout.flush();
    std::cerr << protocol << " iteration " << iteration << ": verified all " << count << " outputs\n";
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise
int main(int argc, char** argv) {
  try {
    const std::string protocol = argc > 1 ? argv[1] : "spdzwise";
    SPU_ENFORCE(protocol == "spdzwise" || protocol == "aby3");
    const auto count = argc > 2 ? std::stoll(argv[2]) : 100000;
    const auto repetitions = argc > 3 ? std::stoi(argv[3]) : 7;
    SPU_ENFORCE(count > 0 && repetitions > 0 && argc <= 4);
    spu::mpc::spdzwise::Run(protocol, count, repetitions);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
