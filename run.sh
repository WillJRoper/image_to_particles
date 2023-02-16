#!/bin/bash

# Run SWIFT
../swiftsim-master/swift --hydro --threads=4 album_cover.yml

python3 makeMovieSwiftsimIO.py

