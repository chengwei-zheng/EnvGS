#!/bin/bash
# gui.sh — open any trained EnvGS run in evc-gui (2026-11-09; replaces calling gui_mirror_b.sh with GROUP / VIEWER / PYTHONPATH by hand).
#
#   gui.sh <run>            run = full name (envgs_ prefix optional), else a unique name ending, else a unique substring
#                           (e.g. oracleall -> ..._scons_oracleall, oracleall_off020, oraclenormmask)
#   gui.sh -l [pattern]     list trained runs (group, time of latest.pt, still training?)
#   gui.sh -n <run>         dry run: print what would be launched
#
# Resolved automatically:
#   group     the configs/exps/envgs/<group>/ that holds envgs_<run>.yaml (must match training, otherwise evc-gui silently loads an
#             untrained model: exp_name is always envgs/<group>/envgs_<run>)
#   viewer    configs/specs/<viewer>.yaml by group / dataset: corridor c2_* -> corridor_c2_viewer, other corridor -> corridor_viewer,
#             mirror_b / mirror_a / bathroom -> <group>_viewer
#   tracer    run name containing _off0NN (START_OFFSET variant) -> PYTHONPATH=build_variants/dst_off0NN/pkg, as in training;
#             otherwise the installed tracer
# Environment overrides: GPU (default 2 = display GPU), VIEWER, TRACER (e.g. off020, or 'installed'), FORCE=1 (open even if another
# viewer is running). One viewer at a time: refuses while another evc-gui is running. Closing the window saves nothing.
set -e
ROOT=/home/yiinqiang/SMBC/EnvGS/EnvGS
cd $ROOT
EXP=configs/exps/envgs

list() {
    printf '%-10s %-12s %-9s %s\n' group latest.pt status run
    for d in data/trained_model/envgs/*/envgs_*${1:+*$1*}; do
        [ -d "$d" ] || continue
        g=$(basename "$(dirname "$d")"); n=$(basename "$d"); n=${n#envgs_}
        e=$(date -r "$d/latest.pt" '+%m-%d %H:%M' 2>/dev/null || true)
        st=done; pgrep -f "evc-train .*exp_name=envgs/$g/envgs_$n( |$)" > /dev/null && st=training
        printf '%-10s %-12s %-9s %s\n' "$g" "${e:--}" "$st" "$n"
    done
}

dry=0
case "$1" in
    -l) list "$2"; exit 0 ;;
    -n) dry=1; shift ;;
    -h|--help|'') sed -n '2,19p' "$0"; exit 0 ;;
esac
q=${1#envgs_}

# resolve the run: exact name first, then unique substring
mapfile -t hits < <(ls $EXP/*/envgs_$q.yaml 2>/dev/null)
[ ${#hits[@]} -eq 0 ] && mapfile -t hits < <(ls $EXP/*/envgs_*$q.yaml 2>/dev/null)
[ ${#hits[@]} -eq 0 ] && mapfile -t hits < <(ls $EXP/*/envgs_*$q*.yaml 2>/dev/null)
if [ ${#hits[@]} -ne 1 ]; then
    echo "run '$q': ${#hits[@]} matching configs" >&2; printf '  %s\n' "${hits[@]}" >&2; exit 1
fi
cfg=${hits[0]}; G=$(basename "$(dirname "$cfg")"); s=$(basename "$cfg" .yaml); s=${s#envgs_}

if [ -z "$VIEWER" ]; then
    case "$G" in
        corridor) [[ $s == c2_* ]] && VIEWER=corridor_c2_viewer || VIEWER=corridor_viewer ;;
        *) VIEWER=${G}_viewer ;;
    esac
fi
[ -f configs/specs/$VIEWER.yaml ] || { echo "no viewer spec configs/specs/$VIEWER.yaml (set VIEWER=...)" >&2; exit 1; }

TM=data/trained_model/envgs/$G/envgs_$s
[ -f $TM/latest.pt ] || { echo "no trained model in $TM" >&2; exit 1; }
ep=$(date -r $TM/latest.pt '+%m-%d %H:%M')
pgrep -f "evc-train .*exp_name=envgs/$G/envgs_$s( |$)" > /dev/null && echo "note: $s is still training: the viewer shows the checkpoint of $ep"

TR=${TRACER:-$(grep -oE '_off0[0-9]+' <<< "$s" | tail -1 | tr -d _)}
if [ -n "$TR" ] && [ "$TR" != installed ]; then
    PKG=$ROOT/build_variants/dst_$TR/pkg
    [ -d $PKG ] || { echo "tracer variant $PKG not found" >&2; exit 1; }
    export PYTHONPATH=$PKG${PYTHONPATH:+:$PYTHONPATH}
fi

echo "run     $G/envgs_$s (latest.pt $ep)"
echo "viewer  configs/specs/$VIEWER.yaml   GPU ${GPU:-2}"
echo "tracer  ${PKG:-installed}"
[ $dry = 1 ] && exit 0

if [ -z "$FORCE" ] && pgrep -f "evc-gui -c" > /dev/null; then
    echo "another viewer is running (one at a time; close it or FORCE=1):" >&2; pgrep -af "evc-gui -c" | cut -c1-200 >&2; exit 1
fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate envgs
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
# Keep IBus (XIM) out of the GLFW window: otherwise WASD typed in the viewer is buffered by the IME as pinyin
export XMODIFIERS=@im=none
python -c "import diff_surfel_tracing as d; print('tracer  loaded from', d.__file__)"
CUDA_VISIBLE_DEVICES=${GPU:-2} exec evc-gui -c $cfg,configs/specs/$VIEWER.yaml exp_name=envgs/$G/envgs_$s < /dev/null
