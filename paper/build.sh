#!/bin/bash
# 双版本编译脚本
# 需要 TeX Live 2026: /usr/local/texlive/2026/bin/x86_64-linux
# TIFS版: main_tifs.tex (IEEEtran) — 12页
# TCHES版: main_tches.tex (iacrj_local.cls) — 22页
#   官方提交时: 把 main_tches.tex 的 \documentclass{iacrj_local} 改回 {iacrj}
#   (官方系统提供 alphaurl.bst, 本地用标准 alpha 样式代替)
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH

build() {
    local name=$1
    pdflatex -interaction=nonstopmode "$name" >/dev/null
    bibtex "$name" >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$name" >/dev/null
    pdflatex -interaction=nonstopmode "$name" | grep -E "^!|Output"
}

echo "=== TIFS (IEEEtran) ==="
build main_tifs
echo "=== TCHES (iacrj) ==="
build main_tches
