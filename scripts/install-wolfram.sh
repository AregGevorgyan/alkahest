#!/usr/bin/env bash
# One-time setup: activate Wolfram Engine for Developers (V1-13 benchmarks).
#
# Prerequisites:
#   1. Install WolframEngine: https://www.wolfram.com/engine/
#   2. Create a free Wolfram ID at https://account.wolfram.com/
#   3. Register your Engine license at https://www.wolfram.com/engine/free-license/
#
# Then run:
#   bash install-wolfram.sh
set -euo pipefail

echo "Activating Wolfram Engine for Developers..."
echo "(You will be prompted for your Wolfram ID activation key and password.)"
wolframscript -activate

echo ""
echo "Testing activation..."
result=$(wolframscript -code "1 + 1" 2>&1)
if echo "$result" | grep -q "^2$"; then
    echo "Wolfram Engine activated successfully!"
    echo "Installing Python client: pip install wolframclient"
    pip install wolframclient
else
    echo "Activation may have failed. Output: $result"
    exit 1
fi
