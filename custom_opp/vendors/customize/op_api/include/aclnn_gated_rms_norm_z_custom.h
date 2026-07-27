
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GATED_RMS_NORM_ZCUSTOM_H_
#define ACLNN_GATED_RMS_NORM_ZCUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGatedRmsNormZCustomGetWorkspaceSize
 * parameters :
 * core : required
 * zSilu : required
 * gamma : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGatedRmsNormZCustomGetWorkspaceSize(
    const aclTensor *core,
    const aclTensor *zSilu,
    const aclTensor *gamma,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGatedRmsNormZCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGatedRmsNormZCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
