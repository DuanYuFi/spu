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

#pragma once

#include "libspu/core/array_ref.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/spdzwisefield/value.h"

namespace spu::mpc::spdzwisefield {

// Reference:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
class BitInject : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "bitinject";

  ce::CExpr latency() const override { return ce::Const(2); }

  // Note: when nbits is large, OT method will be slower then circuit method.
  ce::CExpr comm() const override {
    return 2 * ce::K() * ce::K()  // the OT
           + ce::K()              // partial send
        ;
  }

  // FIXME: bypass unittest.
  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

}  // namespace spu::mpc::spdzwisefield