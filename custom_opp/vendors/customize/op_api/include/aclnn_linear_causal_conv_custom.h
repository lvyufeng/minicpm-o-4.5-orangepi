
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LINEAR_CAUSAL_CONV_CUSTOM_H_
#define ACLNN_LINEAR_CAUSAL_CONV_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLinearCausalConvCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearCausalConvCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLinearCausalConvCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearCausalConvCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
