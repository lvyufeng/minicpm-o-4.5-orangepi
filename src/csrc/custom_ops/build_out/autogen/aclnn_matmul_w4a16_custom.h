
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATMUL_W4A16CUSTOM_H_
#define ACLNN_MATMUL_W4A16CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatmulW4a16CustomGetWorkspaceSize
 * parameters :
 * x : required
 * w : required
 * scales : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulW4a16CustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *w,
    const aclTensor *scales,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatmulW4a16Custom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulW4a16Custom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
