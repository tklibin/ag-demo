#!/usr/bin/env bash
# LightGBM's macOS wheel links against Homebrew's libomp. The usual fix is `brew install libomp`.
# Without Homebrew, this points LightGBM at the libomp that scikit-learn's wheel already bundles.
# It only touches files inside the virtualenv. Re-run after reinstalling lightgbm.
set -euo pipefail

VENV="${1:-.venv}"
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
RPATH="@loader_path/../../sklearn/.dylibs"

if ! otool -l "$LIB" | grep -q "$RPATH"; then
  install_name_tool -add_rpath "$RPATH" "$LIB"
  codesign --force -s - "$LIB"   # editing the binary invalidates its signature; re-sign ad hoc
fi
"$VENV/bin/python" -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'OK')"
