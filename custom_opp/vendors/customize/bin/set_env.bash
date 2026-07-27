#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/mnt/data/minicpm-o-4.5-orangepi/custom_opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/mnt/data/minicpm-o-4.5-orangepi/custom_opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
