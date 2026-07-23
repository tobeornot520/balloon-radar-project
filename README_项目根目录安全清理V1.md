# Balloon Radar Project 根目录安全清理 V1

## 清理范围

本工具只处理两类内容：

1. 项目根目录中已经完成集成的旧补丁、旧诊断与一次性 package 目录；
2. 根目录中的安装器、补丁压缩包、验收压缩包、阶段日志与阶段 README。

不会自动处理以下核心内容：

- `data/`
- `datasets/`
- `models/`
- `features/`
- `training/`
- `scripts/`
- `configs/`
- `results/`
- `checkpoints/`
- `backups/`
- `environment.yml`
- `requirements-lock.txt`

## 安全机制

- 默认 `--preview`，不修改任何文件；
- `--execute` 不会永久删除，而是移动到 `_cleanup_archive/cleanup_时间戳/`；
- 可使用 `--restore-latest` 恢复；
- 缓存文件通过 `--clean-caches` 删除，缓存可自动重新生成；
- 永久删除必须显式提供确认短语，不建议使用。

## 命令

```bash
cd ~/projects/balloon_radar_project

python cleanup_balloon_radar_project_v1.py --preview \
  2>&1 | tee project_cleanup_preview.log

python cleanup_balloon_radar_project_v1.py --execute --clean-caches \
  2>&1 | tee project_cleanup_execute.log
```

恢复最近一次清理：

```bash
python cleanup_balloon_radar_project_v1.py --restore-latest
```
