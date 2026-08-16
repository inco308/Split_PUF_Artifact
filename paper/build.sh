#!/bin/bash
# TCHES版编译 (需要 TeX Live 2026: /usr/local/texlive/2026/bin/x86_64-linux)
# 官方提交时: 把 main.tex 的 \documentclass{iacrj_local} 改回 {iacrj}
# (官方系统提供 alphaurl.bst, 本地用标准 alpha 样式代替)
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH
pdflatex -interaction=nonstopmode main >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main >/dev/null
pdflatex -interaction=nonstopmode main | grep -E "^!|Output"
