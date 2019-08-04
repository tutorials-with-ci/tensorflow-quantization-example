# TensorFlow Quantization Example

[![Build Status](https://travis-ci.com/tutorials-with-ci/tensorflow-quantization-example.svg?branch=master)](https://travis-ci.com/tutorials-with-ci/tensorflow-quantization-example)

TensorFlow Quantization Example, for TensorFlow Lite

(There are still problems and we are looking for a solution.)

## Steps

Same as the steps in the configuration file:

```yml
language: python

python:
    - '3.6'

install:
    - pip install -r requirements.txt

script:
    - python train.py
    - python test.py
    - sh ./freeze.sh
    - sh ./quantization.sh
    - python ./load_tflite.py
```
