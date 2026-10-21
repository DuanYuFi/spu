// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "libspu/mpc/kernel.h"

namespace spu::mpc::spdzwise::ring {

// SPDZ-wise arithmetic over Z/(2^128), with FM64 application semantics.
// Kernel binding names remain the standard SPU arithmetic API names.
class P2A final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "p2a"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class Negate final : public UnaryKernel {
 public:
  static constexpr const char* kBindName() { return "negate_a"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& in) const override;
};

class AddAA final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class AddAP final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "add_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MulAA final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MulAP final : public BinaryKernel {
 public:
  static constexpr const char* kBindName() { return "mul_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatmulAA final : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_aa"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatmulAP final : public MatmulKernel {
 public:
  static constexpr const char* kBindName() { return "mmul_ap"; }
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

void registerArithmeticKernels(Object* object);

}  // namespace spu::mpc::spdzwise::ring
