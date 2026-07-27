
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LINEAR_GATED_DELTA_RULE_CUSTOM_H_
#define ACLNN_LINEAR_GATED_DELTA_RULE_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLinearGatedDeltaRuleCustomGetWorkspaceSize
 * parameters :
 * mixed : required
 * beta : required
 * decay : required
 * scratch : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearGatedDeltaRuleCustomGetWorkspaceSize(
    const aclTensor *mixed,
    const aclTensor *beta,
    const aclTensor *decay,
    const aclTensor *scratch,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLinearGatedDeltaRuleCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLinearGatedDeltaRuleCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
