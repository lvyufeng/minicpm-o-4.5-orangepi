
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LINEAR_CAUSAL_CONV_STEP_CUSTOM_H_
#define ACLNN_LINEAR_CAUSAL_CONV_STEP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLinearCausalConvStepCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weightT : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearCausalConvStepCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightT,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLinearCausalConvStepCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearCausalConvStepCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
