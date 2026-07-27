
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ATTENTION_STEP_CUSTOM_H_
#define ACLNN_ATTENTION_STEP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAttentionStepCustomGetWorkspaceSize
 * parameters :
 * query : required
 * kCache : required
 * vCache : required
 * context : required
 * numQHeads : required
 * numKvHeads : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAttentionStepCustomGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *kCache,
    const aclTensor *vCache,
    int64_t context,
    int64_t numQHeads,
    int64_t numKvHeads,
    double scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAttentionStepCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAttentionStepCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
