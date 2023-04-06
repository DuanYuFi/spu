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
#include "libspu/mpc/beaver/value.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::beaver {
// Reference:
// ABY3: A Mixed Protocol Framework for Machine Learning
// P16 5.3 Share Conversions, Bit Decomposition
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2 + log(nbits) from 2 rotate and 1 ppa.
class A2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2b";

  ce::CExpr latency() const override {
    // 1 * AddBB : log(k) + 1
    // 1 * rotate: 1
    return Log(ce::K()) + 1 + 1;
  }

  // TODO: this depends on the adder circuit.
  ce::CExpr comm() const override {
    // 1 * AddBB : logk * k + k
    // 1 * rotate: k
    return Log(ce::K()) * ce::K() + ce::K() * 2;
  }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class B2AByPPA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "b2a";

  Kind kind() const override { return Kind::Dynamic; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class MsbA2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  ce::CExpr latency() const override {
    // 1 * carry_out : log(k) + 1
    // 1 * rotate: 1
    return Log(ce::K()) + 1 + 1;
  }

  ce::CExpr comm() const override {
    // 1 * carry_out: k + 2 * k
    // 1 * rotate: k
    return ce::K() * 4;
  }

  float getCommTolerance() const override { return 0.2; }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};
}  // namespace spu::mpc::beaver