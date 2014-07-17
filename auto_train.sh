#!/bin/bash

BASEDIR="$(dirname $(readlink -e $0))"

pushd $BASEDIR > /dev/null
./auto_train.py
popd > /dev/null
