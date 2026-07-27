
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_W8A8QUANTIZE_CUSTOM_H_
#define ACLNN_W8A8QUANTIZE_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnW8a8QuantizeCustomGetWorkspaceSize
 * parameters :
 * x : required
 * xqOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW8a8QuantizeCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *xqOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnW8a8QuantizeCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW8a8QuantizeCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
