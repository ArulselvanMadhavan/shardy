#!/bin/bash
bazel run -c opt //shardy/tools:sdy_example -- --show-dialects -o="module.mlir"
