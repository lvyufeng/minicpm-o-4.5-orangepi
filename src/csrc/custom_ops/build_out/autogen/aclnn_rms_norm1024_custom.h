
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMS_NORM1024CUSTOM_H_
#define ACLNN_RMS_NORM1024CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmsNorm1024CustomGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * epsilon : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNorm1024CustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmsNorm1024Custom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNorm1024Custom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
