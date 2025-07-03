#!/bin/bash
# coding=utf-8

if [ -d "build" ]; then
    rm -rf build/*
else
    mkdir build
fi

cp setup_on_mlu.py build/

cd build
python setup_on_mlu.py install
