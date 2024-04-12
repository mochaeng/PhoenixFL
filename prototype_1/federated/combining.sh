#!/bin/bash

relative_path="$(dirname "$0")"
cd "$relative_path/../../"

python -m prototype_1.federated.combining
