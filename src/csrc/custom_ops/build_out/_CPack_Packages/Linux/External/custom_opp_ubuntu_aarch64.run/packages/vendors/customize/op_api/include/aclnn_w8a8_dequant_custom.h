
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_W8A8DEQUANT_CUSTOM_H_
#define ACLNN_W8A8DEQUANT_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnW8a8DequantCustomGetWorkspaceSize
 * parameters :
 * acc : required
 * xScale : required
 * wScale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW8a8DequantCustomGetWorkspaceSize(
    const aclTensor *acc,
    const aclTensor *xScale,
    const aclTensor *wScale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnW8a8DequantCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnW8a8DequantCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
