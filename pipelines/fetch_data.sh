#!/usr/bin/env bash
# Downloads the raw MovieLens data the artifact builder needs.
# ml-20m is required to recover the training sample (the encoder mapping);
# ml-32m supplies richer ratings, tags and TMDB links.
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "$DEST"
for ds in ml-20m ml-32m; do
  if [ ! -d "$DEST/$ds" ]; then
    echo "downloading $ds ..."
    curl -fsSL -o "$DEST/$ds.zip" "https://files.grouplens.org/datasets/movielens/$ds.zip"
    unzip -q "$DEST/$ds.zip" -d "$DEST"
    rm "$DEST/$ds.zip"
  fi
done
echo "done -> $DEST"
