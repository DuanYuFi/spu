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

#include "libspu/mpc/beaver/protocol.h"

#include "libspu/mpc/beaver/arithmetic.h"
#include "libspu/mpc/beaver/boolean.h"
#include "libspu/mpc/beaver/conversion.h"
#include "libspu/mpc/beaver/state.h"
#include "libspu/mpc/beaver/type.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/ab_kernels.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"

namespace spu::mpc {

std::unique_ptr<Object> makeBeaverProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  beaver::registerTypes();
  auto obj =
      std::make_unique<Object>(fmt::format("{}-{}", lctx->Rank(), "BEAVER"));

  // add beaver state
  obj->addState<BeaverState>(lctx);

  obj->addState<Z2kState>(conf.field());

  // add communicator
  obj->addState<Communicator>(lctx);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);

  // register public kernels.
  regPub2kKernels(obj.get());

  // register api kernels
  regABKernels(obj.get());

  // register arithmetic & binary kernels
  obj->regKernel<beaver::P2A>();
  obj->regKernel<beaver::A2P>();
  obj->regKernel<beaver::NotA>();
  obj->regKernel<beaver::RandA>();
  obj->regKernel<beaver::AddAP>();
  obj->regKernel<beaver::AddAA>();
  obj->regKernel<beaver::MulAP>();
  obj->regKernel<beaver::MulAA>();
  // obj->regKernel<beaver::MulA1B>();
  obj->regKernel<beaver::MatMulAP>();
  obj->regKernel<beaver::MatMulAA>();
  obj->regKernel<beaver::LShiftA>();

  obj->regKernel<beaver::TruncA>();

  obj->regKernel<beaver::MsbA2B>();

  obj->regKernel<beaver::CommonTypeB>();
  obj->regKernel<beaver::CastTypeB>();
  obj->regKernel<beaver::B2P>();
  obj->regKernel<beaver::P2B>();
  obj->regKernel<common::AddBB>();
  obj->regKernel<beaver::A2B>();
  // obj->regKernel<beaver::B2ASelector>();
  // obj->regKernel<aby3::B2AByOT>();
  obj->regKernel<beaver::B2AByPPA>();
  obj->regKernel<beaver::AndBP>();
  obj->regKernel<beaver::AndBB>();
  obj->regKernel<beaver::XorBP>();
  obj->regKernel<beaver::XorBB>();
  obj->regKernel<beaver::LShiftB>();
  obj->regKernel<beaver::RShiftB>();
  obj->regKernel<beaver::ARShiftB>();
  obj->regKernel<beaver::BitrevB>();
  obj->regKernel<beaver::BitIntlB>();
  obj->regKernel<beaver::BitDeintlB>();

  return obj;
}
}  // namespace spu::mpc