#!/bin/bash

BASEDIR="$(dirname $(readlink -e $0))"

pushd $BASEDIR > /dev/null
while [ 1 ]; do
    ./auto_train.py
done
popd > /dev/null
