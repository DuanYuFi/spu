
#pragma once

#include "libspu/core/array_ref.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/spdzwisefield/utils.h"

namespace spu::mpc::spdzwisefield {

class P2A : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2a";

  /*
   * Note that verification is needed for checking the correctness
   * of mac, but it is not included in the latency and comm.
   */
  ce::CExpr latency() const override {
    // one for pass around rss share and
    // one for mac_key * share

    return ce::Const(2);
  }

  ce::CExpr comm() const override { return ce::K() * ce::Const(2); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

class A2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2p";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

// Semi-honest P2A
class P2ASH : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2ash";

  /*
   * Note that verification is needed for checking the correctness
   * of mac, but it is not included in the latency and comm.
   */
  ce::CExpr latency() const override {
    // one for pass around rss share and
    // one for mac_key * share

    return ce::Const(1);
  }

  ce::CExpr comm() const override { return ce::K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

// Semi-honest A2P
class A2PSH : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "a2psh";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override;
};

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
class AddAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_aa";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class AddAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
class MulAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_aa";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * ce::Const(2); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ap";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * ce::Const(2); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class MulAASemiHonest : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_aa_sh";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K() * ce::Const(2); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override;
};

class LShiftA : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_a";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;
};
class TruncA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  ce::CExpr latency() const override { return ce::Const(1); }

  ce::CExpr comm() const override { return ce::K(); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override;

  bool hasMsbError() const override { return true; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Random;
  }
};

}  // namespace spu::mpc::spdzwisefield