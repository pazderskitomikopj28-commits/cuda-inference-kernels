#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
compute-sanitizer --tool memcheck "${build_dir}/rmsnorm_tests"
compute-sanitizer --tool racecheck "${build_dir}/rmsnorm_tests"
