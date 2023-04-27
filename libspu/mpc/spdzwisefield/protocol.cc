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

#include "libspu/mpc/spdzwisefield/protocol.h"

#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/ab_kernels.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/spdzwisefield/arithmetic.h"
#include "libspu/mpc/spdzwisefield/boolean.h"
#include "libspu/mpc/spdzwisefield/conversion.h"
#include "libspu/mpc/spdzwisefield/state.h"
#include "libspu/mpc/spdzwisefield/type.h"

// TODO: In open stage for malicious setting, I think it should
// communicate two elements for the consistence.
namespace spu::mpc {

std::unique_ptr<Object> makeSpdzWiseFieldProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  spdzwisefield::registerTypes();

  auto obj = std::make_unique<Object>("SPDZWISEFIELD");

  // add communicator
  obj->addState<Communicator>(lctx);
  obj->addState<Z2kState>(FM64);

  // register random states & kernels.
  obj->addState<PrgState>(lctx);

  obj->addState<BeaverState>(lctx);

  auto* prg_state = obj->getState<PrgState>();
  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  std::array<uint64_t, 2> key;

  key[0] = r0[0];
  key[1] = r1[0];

  obj->addState<SpdzWiseFieldState>(lctx, key, conf.field());

  // register public kernels.
  regPub2kKernels(obj.get());

  // register api kernels
  regABKernels(obj.get());

  // register arithmetic & binary kernels
  obj->regKernel<spdzwisefield::P2A>();
  obj->regKernel<spdzwisefield::A2P>();
  obj->regKernel<spdzwisefield::P2ASH>();
  obj->regKernel<spdzwisefield::A2PSH>();

  obj->regKernel<spdzwisefield::CommonTypeB>();
  obj->regKernel<spdzwisefield::CastTypeB>();
  obj->regKernel<spdzwisefield::B2P>();
  obj->regKernel<spdzwisefield::P2B>();
  obj->regKernel<common::AddBB>();
  // obj->regKernel<spdzwisefield::A2B>();
  // obj->regKernel<spdzwisefield::B2ASelector>();
  // obj->regKernel<aby3::B2AByOT>();
  // obj->regKernel<spdzwisefield::B2AByPPA>();
  obj->regKernel<spdzwisefield::AndBP>();
  obj->regKernel<spdzwisefield::AndBB>();
  obj->regKernel<spdzwisefield::XorBP>();
  obj->regKernel<spdzwisefield::XorBB>();
  obj->regKernel<spdzwisefield::LShiftB>();
  obj->regKernel<spdzwisefield::RShiftB>();
  obj->regKernel<spdzwisefield::ARShiftB>();
  obj->regKernel<spdzwisefield::BitrevB>();
  obj->regKernel<spdzwisefield::BitIntlB>();
  obj->regKernel<spdzwisefield::BitDeintlB>();

  obj->regKernel<spdzwisefield::BitInject>();

  return obj;
}

}  // namespace spu::mpc