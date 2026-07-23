#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# BC-DPG-FCN v3 工程上下文采集脚本
#
# 作用：
#   1. 定位当前雷达工程并采集 BC-DPG/DPG/V4/V3 相关源码
#   2. 自动提取 Python 类、函数、forward/train/evaluate 接口签名
#   3. 记录数据清单、预测表、配置文件的表头和少量样例
#   4. 记录 Git 状态、Python/PyTorch/CUDA 环境及工程目录结构
#   5. 生成一个可上传给 ChatGPT 的压缩包
#
# 默认工程目录：
#   ~/projects/balloon_radar_project
#
# 使用：
#   chmod +x collect_bc_dpg_v3_context.sh
#   ./collect_bc_dpg_v3_context.sh
#
# 指定工程目录：
#   ./collect_bc_dpg_v3_context.sh /path/to/balloon_radar_project
#
# 可选环境变量：
#   OUTPUT_DIR=~/Downloads ./collect_bc_dpg_v3_context.sh
#   INCLUDE_RESULT_SAMPLES=0 ./collect_bc_dpg_v3_context.sh
#
# 安全说明：
#   - 不打包 .mat/.pt/.pth/.ckpt 等原始数据和模型权重
#   - 不打包 .env、SSH 密钥、访问令牌和 Git 凭证
#   - CSV 仅复制候选 manifest/预测/指标文件，并限制文件大小
# ============================================================

PROJECT_ROOT="${1:-$HOME/projects/balloon_radar_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME}"
INCLUDE_RESULT_SAMPLES="${INCLUDE_RESULT_SAMPLES:-1}"
MAX_TEXT_FILE_MB="${MAX_TEXT_FILE_MB:-8}"
MAX_CSV_FILE_MB="${MAX_CSV_FILE_MB:-5}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="bc_dpg_v3_context_${TIMESTAMP}"
WORK_DIR="$(mktemp -d)"
BUNDLE_DIR="${WORK_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

die() {
    echo "[错误] $*" >&2
    exit 1
}

log() {
    echo "[信息] $*"
}

warn() {
    echo "[警告] $*" >&2
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

safe_relpath() {
    python - "$PROJECT_ROOT" "$1" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
path = Path(sys.argv[2]).resolve()
try:
    print(path.relative_to(root))
except ValueError:
    print(path.name)
PY
}

is_sensitive_path() {
    local p="${1,,}"
    case "$p" in
        *"/.git/"*|*/.git) return 0 ;;
        *"/.env"*|*"/env.local"*|*"/credentials"*|*"/credential"*|*"/secrets"*|*"/secret"*|*"/token"*|*"/tokens"*|*"/id_rsa"*|*"/id_ed25519"*|*"/known_hosts"*|*"/.ssh/"*) return 0 ;;
        *) return 1 ;;
    esac
}

is_binary_or_heavy() {
    local p="${1,,}"
    case "$p" in
        *.mat|*.h5|*.hdf5|*.npy|*.npz|*.pt|*.pth|*.ckpt|*.onnx|*.engine|*.bin|*.pkl|*.pickle|*.joblib|*.zip|*.rar|*.7z|*.tar|*.gz|*.bz2|*.xz|*.mp4|*.avi|*.mov|*.mkv|*.png|*.jpg|*.jpeg|*.tif|*.tiff|*.gif|*.pdf|*.doc|*.docx|*.ppt|*.pptx) return 0 ;;
        *) return 1 ;;
    esac
}

copy_text_file() {
    local src="$1"
    local group="$2"

    [[ -f "$src" ]] || return 0
    is_sensitive_path "$src" && return 0
    is_binary_or_heavy "$src" && return 0

    local size_mb
    size_mb="$(du -m "$src" 2>/dev/null | awk '{print $1}' || echo 999)"
    if [[ "$size_mb" -gt "$MAX_TEXT_FILE_MB" ]]; then
        printf '%s\t%s MB\t%s\n' "SKIPPED_LARGE_TEXT" "$size_mb" "$src" >> "${BUNDLE_DIR}/reports/skipped_files.tsv"
        return 0
    fi

    local rel
    rel="$(safe_relpath "$src")"
    local dst="${BUNDLE_DIR}/source/${group}/${rel}"
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
}

copy_csv_candidate() {
    local src="$1"
    local group="$2"

    [[ -f "$src" ]] || return 0
    is_sensitive_path "$src" && return 0

    local size_mb
    size_mb="$(du -m "$src" 2>/dev/null | awk '{print $1}' || echo 999)"
    if [[ "$size_mb" -gt "$MAX_CSV_FILE_MB" ]]; then
        printf '%s\t%s MB\t%s\n' "SKIPPED_LARGE_CSV" "$size_mb" "$src" >> "${BUNDLE_DIR}/reports/skipped_files.tsv"
        return 0
    fi

    local rel
    rel="$(safe_relpath "$src")"
    local dst="${BUNDLE_DIR}/tables/${group}/${rel}"
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
}

[[ -d "$PROJECT_ROOT" ]] || die "工程目录不存在：$PROJECT_ROOT"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
mkdir -p "$OUTPUT_DIR" || die "无法创建输出目录：$OUTPUT_DIR"

mkdir -p \
    "${BUNDLE_DIR}/source" \
    "${BUNDLE_DIR}/tables" \
    "${BUNDLE_DIR}/reports" \
    "${BUNDLE_DIR}/metadata"

log "工程目录：$PROJECT_ROOT"
log "采集目录：$BUNDLE_DIR"

# ------------------------------------------------------------
# 1. 基本说明
# ------------------------------------------------------------
cat > "${BUNDLE_DIR}/README_上传说明.md" <<EOF
# BC-DPG-FCN v3 工程上下文包

生成时间：$(date -Iseconds)
工程根目录：\`$PROJECT_ROOT\`

## 该压缩包用于确认

1. 当前 BC-DPG-FCN v3、DPG-FCN、数据集和损失函数的真实文件位置；
2. 模型类、\`forward()\`、训练、验证和评估函数的实际签名；
3. background branch、scan context、target protection、suppression loss 的实现位置；
4. V4 六折 manifest 的字段；
5. validation/test prediction CSV 的字段；
6. checkpoint 的键名和 state_dict 层名；
7. 当前训练命令、配置和结果目录命名方式；
8. 后续消融脚本应采用的最小改动接口。

## 已主动排除

- 原始雷达数据：\`.mat\`、\`.h5\`、\`.npy\`、\`.npz\`
- 模型权重：\`.pt\`、\`.pth\`、\`.ckpt\`
- 图片、视频、压缩包和 Office/PDF 文件
- \`.env\`、SSH 密钥、凭证、token、secret 等敏感文件
- 整个 \`.git\` 对象数据库

## 上传前建议检查

执行：

\`\`\`bash
tar -tzf "$ARCHIVE_PATH" | less
\`\`\`

确认压缩包中不存在不希望分享的内容后再上传。
EOF

# ------------------------------------------------------------
# 2. 环境、Git 和目录结构
# ------------------------------------------------------------
{
    echo "generated_at=$(date -Iseconds)"
    echo "project_root=$PROJECT_ROOT"
    echo "hostname=$(hostname 2>/dev/null || true)"
    echo "user=$(id -un 2>/dev/null || true)"
    echo "kernel=$(uname -a 2>/dev/null || true)"
    echo "shell=${SHELL:-unknown}"
} > "${BUNDLE_DIR}/metadata/system.txt"

{
    echo "=== Python ==="
    python --version 2>&1 || true
    command -v python || true
    echo
    echo "=== Conda ==="
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
    conda --version 2>&1 || true
    echo
    echo "=== GPU ==="
    nvidia-smi 2>&1 || true
    echo
    echo "=== PyTorch ==="
    python - <<'PY' 2>&1 || true
try:
    import torch
    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)
    print("cudnn_version:", torch.backends.cudnn.version())
    print("gpu_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"gpu_{i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("PyTorch inspection failed:", repr(exc))
PY
    echo
    echo "=== Key packages ==="
    python -m pip freeze 2>/dev/null | grep -Ei \
        '^(torch|torchvision|torchaudio|numpy|scipy|pandas|matplotlib|scikit|sklearn|h5py|pyyaml|tqdm|tensorboard|tabulate|opencv|einops|seaborn)==' \
        || true
} > "${BUNDLE_DIR}/metadata/environment.txt"

if [[ -d "${PROJECT_ROOT}/.git" ]] && command_exists git; then
    (
        cd "$PROJECT_ROOT"
        {
            echo "=== git status --short ==="
            git status --short || true
            echo
            echo "=== git branch ==="
            git branch --show-current || true
            echo
            echo "=== git log -10 ==="
            git log -10 --oneline --decorate || true
            echo
            echo "=== git remote -v（URL 可能含用户名，但不会读取凭证）==="
            git remote -v 2>/dev/null | sed -E \
                's#(https?://)[^/@:]+:[^/@]+@#\1***:***@#g' || true
            echo
            echo "=== staged diff stat ==="
            git diff --staged --stat || true
            echo
            echo "=== unstaged diff stat ==="
            git diff --stat || true
        }
    ) > "${BUNDLE_DIR}/metadata/git_state.txt"
else
    echo "当前目录不是 Git 仓库，或系统未安装 git。" > "${BUNDLE_DIR}/metadata/git_state.txt"
fi

if command_exists tree; then
    (
        cd "$PROJECT_ROOT"
        tree -a -L 4 \
            -I '.git|__pycache__|.pytest_cache|.mypy_cache|.idea|.vscode|wandb|tensorboard|data/raw|data/processed|checkpoints|weights|*.mat|*.pt|*.pth|*.ckpt'
    ) > "${BUNDLE_DIR}/reports/project_tree.txt" 2>&1 || true
else
    (
        cd "$PROJECT_ROOT"
        find . -maxdepth 4 \
            \( -path './.git' -o -path './data/raw' -o -path './data/processed' -o -path '*/__pycache__' -o -path './checkpoints' -o -path './weights' \) -prune \
            -o -print | sort
    ) > "${BUNDLE_DIR}/reports/project_tree.txt"
fi

# ------------------------------------------------------------
# 3. 搜索关键实现位置
# ------------------------------------------------------------
KEY_PATTERN='BC[-_ ]?DPG|BC_DPG|BCDPG|DPG[-_ ]?FCN|DualPolar|dual[_ -]?branch|background[_ -]?(branch|head|prob|context)|scan[_ -]?(group|context|embedding)|target[_ -]?protection|suppression[_ -]?(loss|shift|gate)|corrected[_ -]?(score|logit)|p_background|bg_prob|fold[_ -]?[1-6]|manifest_v4|cross.?validation'

if command_exists rg; then
    (
        cd "$PROJECT_ROOT"
        rg -n -i \
            --glob '*.py' --glob '*.yaml' --glob '*.yml' --glob '*.json' \
            --glob '*.toml' --glob '*.ini' --glob '*.cfg' --glob '*.md' \
            --glob '!**/.git/**' --glob '!**/__pycache__/**' \
            --glob '!data/raw/**' --glob '!data/processed/**' \
            --glob '!**/*.pt' --glob '!**/*.pth' --glob '!**/*.ckpt' \
            "$KEY_PATTERN" .
    ) > "${BUNDLE_DIR}/reports/key_symbol_search.txt" 2>&1 || true
else
    (
        cd "$PROJECT_ROOT"
        grep -RniE "$KEY_PATTERN" \
            --include='*.py' --include='*.yaml' --include='*.yml' \
            --include='*.json' --include='*.toml' --include='*.ini' \
            --include='*.cfg' --include='*.md' \
            --exclude-dir=.git --exclude-dir=__pycache__ \
            --exclude-dir=raw --exclude-dir=processed .
    ) > "${BUNDLE_DIR}/reports/key_symbol_search.txt" 2>&1 || true
fi

# 将所有命中关键符号的轻量文本源文件复制进 source/key_hits
python - "$PROJECT_ROOT" "${BUNDLE_DIR}/reports/key_symbol_search.txt" "${BUNDLE_DIR}/reports/key_hit_files.txt" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
search_file = Path(sys.argv[2])
output_file = Path(sys.argv[3])

paths = set()
if search_file.exists():
    for line in search_file.read_text(errors="ignore").splitlines():
        # rg/grep 的典型格式：./path/file.py:line:content
        match = re.match(r"^(.*?):\d+:", line)
        if not match:
            continue
        raw = match.group(1)
        p = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            continue
        if p.is_file():
            paths.add(p)

output_file.write_text("\n".join(str(p) for p in sorted(paths)) + ("\n" if paths else ""))
PY

while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    copy_text_file "$file" "key_hits"
done < "${BUNDLE_DIR}/reports/key_hit_files.txt"

# ------------------------------------------------------------
# 4. 复制关键目录中的源码与配置
# ------------------------------------------------------------
CANDIDATE_DIRS=(
    "models"
    "model"
    "networks"
    "datasets"
    "data"
    "training"
    "train"
    "scripts"
    "losses"
    "loss"
    "metrics"
    "evaluation"
    "evaluate"
    "utils"
    "configs"
    "config"
)

for rel_dir in "${CANDIDATE_DIRS[@]}"; do
    abs_dir="${PROJECT_ROOT}/${rel_dir}"
    [[ -d "$abs_dir" ]] || continue

    while IFS= read -r -d '' file; do
        case "${file,,}" in
            *.py|*.yaml|*.yml|*.json|*.toml|*.ini|*.cfg|*.md|*.txt|*.sh)
                copy_text_file "$file" "project_files"
                ;;
        esac
    done < <(
        find "$abs_dir" -type f -print0 \
            -not -path '*/__pycache__/*' \
            -not -path '*/.pytest_cache/*' \
            -not -path '*/raw/*' \
            -not -path '*/processed/*' \
            -not -path '*/weights/*' \
            -not -path '*/checkpoints/*' \
            -not -path '*/wandb/*'
    )
done

# 根目录的轻量配置和说明
while IFS= read -r -d '' file; do
    copy_text_file "$file" "root_files"
done < <(
    find "$PROJECT_ROOT" -maxdepth 1 -type f -print0 | \
    while IFS= read -r -d '' f; do
        case "${f,,}" in
            *.py|*.yaml|*.yml|*.json|*.toml|*.ini|*.cfg|*.md|*.txt|*.sh|requirements*.txt|environment*.yml)
                printf '%s\0' "$f"
                ;;
        esac
    done
)

# ------------------------------------------------------------
# 5. Python AST 接口提取
# ------------------------------------------------------------
python - "$PROJECT_ROOT" "${BUNDLE_DIR}/reports/python_interfaces.txt" "${BUNDLE_DIR}/reports/python_interfaces.json" <<'PY'
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
txt_out = Path(sys.argv[2])
json_out = Path(sys.argv[3])

skip_parts = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea",
    "data", "raw", "processed", "weights", "checkpoints", "wandb",
}
interest = (
    "dpg", "bc_", "background", "suppression", "protect",
    "scan", "fold", "detection", "dataset", "loss", "train", "eval",
)

def annotation_text(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None

def default_text(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"

def signature(node):
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = []

    for arg, default in zip(positional, defaults):
        item = arg.arg
        ann = annotation_text(arg.annotation)
        if ann:
            item += f": {ann}"
        if default is not None:
            item += f" = {default_text(default)}"
        parts.append(item)

    if args.vararg:
        item = "*" + args.vararg.arg
        ann = annotation_text(args.vararg.annotation)
        if ann:
            item += f": {ann}"
        parts.append(item)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        item = arg.arg
        ann = annotation_text(arg.annotation)
        if ann:
            item += f": {ann}"
        if default is not None:
            item += f" = {default_text(default)}"
        parts.append(item)

    if args.kwarg:
        item = "**" + args.kwarg.arg
        ann = annotation_text(args.kwarg.annotation)
        if ann:
            item += f": {ann}"
        parts.append(item)

    result = "(" + ", ".join(parts) + ")"
    returns = annotation_text(node.returns)
    if returns:
        result += f" -> {returns}"
    return result

records = []
for path in sorted(root.rglob("*.py")):
    try:
        rel = path.relative_to(root)
    except ValueError:
        continue

    lower_parts = {p.lower() for p in rel.parts}
    if lower_parts & skip_parts:
        continue

    rel_text = str(rel).lower()
    # 保留所有核心目录；其他目录仅保留名称与任务相关的 Python 文件
    core_dir = rel.parts and rel.parts[0].lower() in {
        "models", "model", "networks", "datasets", "training", "train",
        "scripts", "losses", "loss", "metrics", "evaluation", "evaluate", "utils",
    }
    if not core_dir and not any(token in rel_text for token in interest):
        continue

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source, filename=str(rel))
    except SyntaxError as exc:
        records.append({
            "file": str(rel),
            "parse_error": f"{exc.msg} at line {exc.lineno}",
        })
        continue

    file_rec = {"file": str(rel), "classes": [], "functions": []}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            file_rec["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "signature": signature(node),
                "doc": ast.get_docstring(node),
            })
        elif isinstance(node, ast.ClassDef):
            cls = {
                "name": node.name,
                "line": node.lineno,
                "bases": [annotation_text(base) for base in node.bases],
                "doc": ast.get_docstring(node),
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls["methods"].append({
                        "name": item.name,
                        "line": item.lineno,
                        "signature": signature(item),
                        "doc": ast.get_docstring(item),
                    })
            file_rec["classes"].append(cls)

    if file_rec["classes"] or file_rec["functions"]:
        records.append(file_rec)

json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

lines = []
for rec in records:
    lines.append("=" * 90)
    lines.append(rec["file"])
    if "parse_error" in rec:
        lines.append("PARSE ERROR: " + rec["parse_error"])
        continue
    for func in rec.get("functions", []):
        lines.append(f"  function L{func['line']}: {func['name']}{func['signature']}")
    for cls in rec.get("classes", []):
        bases = ", ".join(x for x in cls["bases"] if x) or "object"
        lines.append(f"  class L{cls['line']}: {cls['name']}({bases})")
        for method in cls["methods"]:
            marker = "  <-- 重点" if method["name"] in {
                "__init__", "forward", "training_step", "validation_step",
                "test_step", "compute_loss", "evaluate", "predict",
            } else ""
            lines.append(
                f"    method L{method['line']}: "
                f"{method['name']}{method['signature']}{marker}"
            )

txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# ------------------------------------------------------------
# 6. 提取 argparse / 配置项 / main 入口
# ------------------------------------------------------------
python - "$PROJECT_ROOT" "${BUNDLE_DIR}/reports/cli_and_config_hints.txt" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])

skip = {".git", "__pycache__", "data", "raw", "processed", "weights", "checkpoints", "wandb"}
patterns = [
    re.compile(r"add_argument\s*\("),
    re.compile(r"ArgumentParser\s*\("),
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"),
    re.compile(r"(?:yaml|json)\.(?:safe_load|load)\s*\("),
    re.compile(r"OmegaConf|Hydra|dataclass|Config"),
]

lines = []
for path in sorted(root.rglob("*.py")):
    try:
        rel = path.relative_to(root)
    except ValueError:
        continue
    if any(part.lower() in skip for part in rel.parts):
        continue
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        content = path.read_text(errors="ignore")
    selected = []
    for lineno, line in enumerate(content.splitlines(), 1):
        if any(p.search(line) for p in patterns):
            selected.append(f"{lineno}: {line.rstrip()}")
    if selected:
        lines.append("=" * 90)
        lines.append(str(rel))
        lines.extend(selected)

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# ------------------------------------------------------------
# 7. Manifest、预测表和指标文件
# ------------------------------------------------------------
python - "$PROJECT_ROOT" "${BUNDLE_DIR}/reports/table_inventory.tsv" "${BUNDLE_DIR}/reports/table_headers_and_samples.txt" <<'PY'
from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import sys

root = Path(sys.argv[1]).resolve()
inventory_out = Path(sys.argv[2])
sample_out = Path(sys.argv[3])

keywords = (
    "manifest", "prediction", "predictions", "metric", "metrics", "summary",
    "fold", "ablation", "rescue", "false_alarm", "background", "threshold",
)
skip_parts = {".git", "__pycache__", "raw", "processed", "weights", "checkpoints", "wandb"}

rows = []
samples = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    try:
        rel = path.relative_to(root)
    except ValueError:
        continue
    if any(part.lower() in skip_parts for part in rel.parts):
        continue

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml"}:
        continue

    rel_lower = str(rel).lower()
    if not any(k in rel_lower for k in keywords):
        continue

    size = path.stat().st_size
    rows.append((str(rel), suffix, size))

    samples.append("=" * 100)
    samples.append(f"FILE: {rel}")
    samples.append(f"SIZE_BYTES: {size}")

    try:
        if suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                for idx, row in enumerate(reader):
                    samples.append(f"ROW_{idx}: " + json.dumps(row, ensure_ascii=False))
                    if idx >= 3:
                        break
        elif suffix in {".json", ".jsonl"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            samples.extend(text.splitlines()[:60])
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            samples.extend(text.splitlines()[:80])
    except Exception as exc:
        samples.append(f"READ_ERROR: {exc!r}")

inventory_out.write_text(
    "relative_path\textension\tsize_bytes\n" +
    "\n".join(f"{p}\t{s}\t{n}" for p, s, n in rows) +
    ("\n" if rows else ""),
    encoding="utf-8",
)
sample_out.write_text("\n".join(samples) + "\n", encoding="utf-8")
PY

if [[ "$INCLUDE_RESULT_SAMPLES" == "1" ]]; then
    while IFS=$'\t' read -r rel suffix size; do
        [[ "$rel" == "relative_path" ]] && continue
        [[ -n "$rel" ]] || continue
        copy_csv_candidate "${PROJECT_ROOT}/${rel}" "candidate_tables"
    done < "${BUNDLE_DIR}/reports/table_inventory.tsv"
fi

# ------------------------------------------------------------
# 8. Checkpoint 元数据（不复制权重）
# ------------------------------------------------------------
find "$PROJECT_ROOT" -type f \
    \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' \) \
    -not -path '*/.git/*' \
    -printf '%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null \
    | sort > "${BUNDLE_DIR}/reports/checkpoint_inventory.tsv" || true

python - "$PROJECT_ROOT" "${BUNDLE_DIR}/reports/checkpoint_inventory.tsv" "${BUNDLE_DIR}/reports/checkpoint_metadata.txt" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
inventory = Path(sys.argv[2])
output = Path(sys.argv[3])

try:
    import torch
except Exception as exc:
    output.write_text(f"无法导入 torch：{exc!r}\n", encoding="utf-8")
    raise SystemExit(0)

entries = []
if inventory.exists():
    for line in inventory.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        path_text = line.split("\t", 1)[0]
        p = Path(path_text)
        name = p.name.lower()
        full = str(p).lower()
        score = 0
        for token, weight in [
            ("bc", 5), ("dpg", 5), ("v3", 4), ("fold", 3),
            ("best", 3), ("seed42", 2), ("final", 1),
        ]:
            if token in full:
                score += weight
        entries.append((score, p))

entries.sort(key=lambda x: (-x[0], str(x[1])))
selected = [p for score, p in entries if score > 0][:12]

lines = [
    "说明：仅使用 torch.load(..., weights_only=True) 读取候选 checkpoint 元数据；",
    "不复制 checkpoint，不输出参数数值，仅输出顶层键和 state_dict 层名。",
    "",
]

for path in selected:
    lines.append("=" * 100)
    try:
        rel = path.resolve().relative_to(root)
    except Exception:
        rel = path.name
    lines.append(f"FILE: {rel}")
    lines.append(f"SIZE_BYTES: {path.stat().st_size}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        lines.append("SKIPPED: 当前 PyTorch 不支持 weights_only=True，为安全起见未加载。")
        continue
    except Exception as exc:
        lines.append(f"LOAD_ERROR: {exc!r}")
        continue

    if isinstance(obj, dict):
        lines.append("TOP_LEVEL_KEYS:")
        for key in list(obj.keys())[:100]:
            value = obj[key]
            desc = type(value).__name__
            if hasattr(value, "shape"):
                desc += f" shape={tuple(value.shape)}"
            lines.append(f"  - {key}: {desc}")

        state = None
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            value = obj.get(key)
            if isinstance(value, dict):
                state = value
                lines.append(f"STATE_DICT_SOURCE: {key}")
                break
        if state is None and all(isinstance(k, str) for k in obj.keys()):
            tensor_like = [v for v in obj.values() if hasattr(v, "shape")]
            if tensor_like and len(tensor_like) >= max(1, len(obj) // 2):
                state = obj
                lines.append("STATE_DICT_SOURCE: top_level")

        if isinstance(state, dict):
            lines.append("STATE_DICT_KEYS:")
            for key in list(state.keys())[:300]:
                value = state[key]
                shape = tuple(value.shape) if hasattr(value, "shape") else None
                lines.append(f"  - {key}: shape={shape}")
    else:
        lines.append(f"OBJECT_TYPE: {type(obj).__name__}")

output.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# ------------------------------------------------------------
# 9. 生成源文件 SHA256，便于确认版本
# ------------------------------------------------------------
(
    cd "$BUNDLE_DIR"
    find source tables reports metadata -type f -print0 | sort -z | \
        xargs -0 sha256sum
) > "${BUNDLE_DIR}/SHA256SUMS.txt" 2>/dev/null || true

# ------------------------------------------------------------
# 10. 最终安全检查与压缩
# ------------------------------------------------------------
if find "$BUNDLE_DIR" -type f | grep -Eiq \
    '(^|/)(\.env|id_rsa|id_ed25519|credentials?|secrets?|tokens?)(\.|$|/)'; then
    die "安全检查失败：采集包中发现疑似敏感文件名。"
fi

if find "$BUNDLE_DIR" -type f | grep -Eiq \
    '\.(mat|pt|pth|ckpt|onnx|h5|hdf5|npy|npz|pkl|pickle)$'; then
    die "安全检查失败：采集包中发现原始数据或权重文件。"
fi

mkdir -p "$OUTPUT_DIR"
tar -C "$WORK_DIR" -czf "$ARCHIVE_PATH" "$BUNDLE_NAME"

log "采集完成。"
echo
echo "压缩包：$ARCHIVE_PATH"
echo "大小：$(du -h "$ARCHIVE_PATH" | awk '{print $1}')"
echo
echo "上传前可检查："
echo "  tar -tzf \"$ARCHIVE_PATH\" | less"
echo
echo "然后将该 .tar.gz 文件上传。"
