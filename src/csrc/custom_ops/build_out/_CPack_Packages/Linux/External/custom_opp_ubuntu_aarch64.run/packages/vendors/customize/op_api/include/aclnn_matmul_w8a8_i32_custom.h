
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATMUL_W8A8I32CUSTOM_H_
#define ACLNN_MATMUL_W8A8I32CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatmulW8a8I32CustomGetWorkspaceSize
 * parameters :
 * x : required
 * w : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulW8a8I32CustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *w,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatmulW8a8I32Custom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulW8a8I32Custom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
