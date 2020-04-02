#!/bin/bash

target="AutoDL_sample_code_submission"
current=`pwd`
if [[ $current =~ $target ]]
then
  pushd ./
else
  pushd AutoDL_sample_code_submission
fi
rm submission.zip
zip -r submission.zip * -x "__pycache__/*" -x "*.sh"
popd
