#!/bin/bash

Config1="options/train/underwater/swin_UEIB.yml"
Config2="options/train/underwater/mambav2_lightSR_UEIB.yml"

python basicsr/train.py -opt $Config2