#!/bin/sh
# Explicit dependency acquisition/build step. Demo/benchmark/evaluation stay offline.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
release_dir="$root/.release"
command -v uv >/dev/null
mkdir -p "$release_dir/wheels"
uv build --wheel --out-dir "$release_dir/wheels" "$root/qedge-common"
uv build --wheel --out-dir "$release_dir/wheels" "$root/edge-rag-demo"
if [ ! -x "$release_dir/venv/bin/python" ]; then
    uv venv --python 3.11 "$release_dir/venv"
fi
uv pip install --python "$release_dir/venv/bin/python" \
    "$release_dir/wheels/qedge_common-0.1.0-py3-none-any.whl" \
    "$release_dir/wheels/edge_rag_demo-0.2.0-py3-none-any.whl"
"$release_dir/venv/bin/python" -m compileall -q "$release_dir/venv/lib"
printf 'Installed. Run %s/venv/bin/qedge-portfolio demo\n' "$release_dir"
