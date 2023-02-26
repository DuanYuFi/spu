// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/spdzwisefield/arithmetic.h"

#include <functional>
#include <future>

#include "spdlog/spdlog.h"

#include "libspu/core/array_ref.h"
#include "libspu/core/trace.h"
// #include "libspu/mpc/aby3/ot.h"
#include "libspu/mpc/spdzwisefield/type.h"
// #include "libspu/mpc/aby3/value.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/utils/linalg.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::spdzwisefield {

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);
  auto* comm = ctx->getState<Communicator>();
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using AShrT = ring2k_t;
    using PShrT = ring2k_t;

    ArrayRef out(makeType<Pub2kTy>(field), in.numel());
    auto _in = ArrayView<std::array<AShrT, 4>>(in);
    auto _out = ArrayView<PShrT>(out);

    std::vector<AShrT> x2(in.numel());

    pforeach(0, in.numel(), [&](int64_t idx) {  //
      x2[idx] = _in[idx][1];
    });

    auto x3 = comm->rotate<AShrT>(x2, "a2p");

    pforeach(0, in.numel(), [&](int64_t idx) {
      _out[idx] = _in[idx][0] + _in[idx][1] + x3[idx];
    });

    return out;
  });
}

}  // namespace spu::mpc::spdzwisefield