#!/bin/bash
# download_data.sh — pull the Rebrickable CSV catalog (updated daily upstream).
# Source of truth for minifig.py. Do NOT commit the CSVs; re-run this to refresh.
# Note: rebrickable.com HTML pages sit behind a Cloudflare challenge, but the
# /media/downloads/*.csv.gz media paths serve directly with a browser UA.
set -e
cd "$(dirname "$0")/data"
BASE=https://rebrickable.com/media/downloads
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
for f in parts colors inventories inventory_parts inventory_sets inventory_minifigs elements minifigs sets; do
    if [ ! -f "${f}.csv" ]; then
        curl -sf -A "$UA" -O "$BASE/$f.csv.gz" && gunzip -f "$f.csv.gz" || echo "skip $f"
    fi
done
echo "catalog files present:"; ls *.csv | tr '\n' ' '; echo
