#!/usr/bin/env bash
# Download the editing datasets into data/ and check their sha256.
#   bash scripts/download_data.sh          # CounterFact (all main experiments)
#   bash scripts/download_data.sh --zsre   # also ZsRE (only for --dataset zsre)
# Both files come from the ROME release (https://rome.baulab.info/data/dsets/).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {  # fetch <url> <destination> <sha256>
    local url=$1 dest=$2 sum=$3
    if [ ! -f "$dest" ]; then
        curl -fL --retry 3 -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
    echo "$sum  $dest" | sha256sum -c -
}

fetch https://rome.baulab.info/data/dsets/counterfact.json data/counterfact.json \
    d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f

if [ "${1:-}" = "--zsre" ]; then
    fetch https://rome.baulab.info/data/dsets/zsre_mend_eval.json data/zsre_mend_eval.json \
        8a371d512f8a6ab175db4ef672181d5c0e52da23298cb51e5a2eeea2f89aa9cd
fi
