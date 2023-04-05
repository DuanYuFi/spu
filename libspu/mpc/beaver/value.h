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
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::beaver {
ArrayRef getShare(const ArrayRef& in, int64_t share_idx);

ArrayRef getFirstShare(const ArrayRef& in);

ArrayRef getSecondShare(const ArrayRef& in);

ArrayRef makeAShare(const ArrayRef& s1, const ArrayRef& s2, FieldType field,
                    int owner_rank = -1);

PtType calcBShareBacktype(size_t nbits);
}  // namespace spu::mpc::beaver