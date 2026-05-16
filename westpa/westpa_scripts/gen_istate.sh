#!/bin/bash
# Convert a basis state restart into an initial state restart for the propagator.
set -e
cp "$WEST_BSTATE_DATA_REF" "$WEST_ISTATE_DATA_REF"
