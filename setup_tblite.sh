meson setup external/tblite/_build external/tblite -Dpython=true --prefix=$PWD/.venv
meson compile -C external/tblite/_build
meson install -C external/tblite/_build

export LD_PRELOAD=/usr/lib/libgfortran.so.5
