#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AddCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(y, ge::TensorType::ALL())
    .OUTPUT(z, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(AddCustom);

REG_OP(AttentionStepCustom)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(k_cache, ge::TensorType::ALL())
    .INPUT(v_cache, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .REQUIRED_ATTR(context, Int)
    .REQUIRED_ATTR(num_q_heads, Int)
    .REQUIRED_ATTR(num_kv_heads, Int)
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(AttentionStepCustom);

REG_OP(GatedRmsNormZCustom)
    .INPUT(core, ge::TensorType::ALL())
    .INPUT(z_silu, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GatedRmsNormZCustom);

REG_OP(LinearCausalConvCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LinearCausalConvCustom);

REG_OP(LinearCausalConvStepCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight_t, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LinearCausalConvStepCustom);

REG_OP(LinearGatedDeltaRuleCustom)
    .INPUT(mixed, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .INPUT(decay, ge::TensorType::ALL())
    .INPUT(scratch, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LinearGatedDeltaRuleCustom);

REG_OP(LinearGatedDeltaRuleStepCustom)
    .INPUT(mixed, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .INPUT(decay, ge::TensorType::ALL())
    .INPUT(state, ge::TensorType::ALL())
    .INPUT(scratch, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LinearGatedDeltaRuleStepCustom);

REG_OP(MatmulCubeCustom)
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .OUTPUT(c, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulCubeCustom);

REG_OP(MatmulVecCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulVecCustom);

REG_OP(MatmulW4a16Custom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .INPUT(scales, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulW4a16Custom);

REG_OP(MatmulW8a8I32Custom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulW8a8I32Custom);

REG_OP(RmsNorm1024Custom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 1e-06)
    .OP_END_FACTORY_REG(RmsNorm1024Custom);

REG_OP(SiluMulCustom)
    .INPUT(gate, ge::TensorType::ALL())
    .INPUT(up, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(SiluMulCustom);

REG_OP(W8a8DequantCustom)
    .INPUT(acc, ge::TensorType::ALL())
    .INPUT(xScale, ge::TensorType::ALL())
    .INPUT(wScale, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(W8a8DequantCustom);

REG_OP(W8a8QuantizeCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(xq, ge::TensorType::ALL())
    .OUTPUT(scale, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(W8a8QuantizeCustom);

}

#endif
