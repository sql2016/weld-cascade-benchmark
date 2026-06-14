# 公开仓库与 Zenodo DOI 发布指南

**目标：** 投稿 *Sensors* 前提供可点击的 GitHub Release URL；Zenodo DOI 可在投稿时或接收后补登记。

---

## 一键生成本地发布包

```powershell
cd F:\AutoResearchClaw-latest-src
.\.venv\Scripts\python.exe scripts\prepare_weld_public_repo.py
.\.venv\Scripts\python.exe scripts\pack_weld_supplementary.py
.\.venv\Scripts\python.exe scripts\render_paper_weld_two_stage_sensors.py
```

生成物：

| 路径 | 用途 |
|------|------|
| `deliverables/public-release/weld-cascade-benchmark/` | 可直接 push 的 GitHub 仓库目录 |
| `deliverables/public-release/weld-cascade-benchmark-v1.0-sensors.zip` | GitHub Release / Zenodo 上传包 |
| `deliverables/WeldCascade_Sensors_Supplementary.zip` | MDPI Susy 补充材料 |
| `deliverables/repository_config.json` | 稿内 Data Availability 读取的 URL/DOI |

---

## Step 1 — 创建 GitHub 公开仓库

推荐仓库名：`weld-cascade-benchmark`（账号 [sql2016](https://github.com/sql2016)）。

```powershell
cd F:\AutoResearchClaw-latest-src\artifacts\rc-weld-two-stage\deliverables\public-release\weld-cascade-benchmark

git init
git add .
git commit -m "WeldCascade benchmark v1.0-sensors (Steel Pipe test-split artefacts)"

# 在 GitHub 网页新建空仓库后：
git remote add origin https://github.com/YOUR_ORG/weld-cascade-benchmark.git
git branch -M main
git push -u origin main
```

### 创建 Release（投稿前必做）

1. GitHub → **Releases** → **Draft a new release**
2. Tag: `v1.0-sensors`
3. Title: `WeldCascade benchmark v1.0-sensors (Sensors submission)`
4. 附件：上传 `weld-cascade-benchmark-v1.0-sensors.zip`
5. 复制 Release URL，例如：  
   `https://github.com/sql2016/weld-cascade-benchmark/releases/tag/v1.0-sensors`

### 更新配置并重编译稿件

编辑 `deliverables/repository_config.json`：

```json
{
  "github_org": "YOUR_ORG",
  "github_repo": "weld-cascade-benchmark",
  "github_url": "https://github.com/YOUR_ORG/weld-cascade-benchmark",
  "release_tag": "v1.0-sensors",
  "release_url": "https://github.com/YOUR_ORG/weld-cascade-benchmark/releases/tag/v1.0-sensors",
  "zenodo_doi": null,
  "zenodo_url": null,
  "version": "1.0.0-sensors"
}
```

然后重跑 `render_paper_weld_two_stage_sensors.py`，Data Availability 会自动写入 Release URL。

---

## Step 2 — Zenodo DOI（推荐，约 10 分钟）

1. 登录 https://zenodo.org（可用 GitHub 账号）
2. **Upload** → 上传 `weld-cascade-benchmark-v1.0-sensors.zip`
3. 填写元数据（`.zenodo.json` 已预填，可对照 `public-release/weld-cascade-benchmark/.zenodo.json`）：
   - **Title:** WeldCascade reproducible benchmark artefacts (Sensors)
   - **Upload type:** Software
   - **Authors:** Zhu, Yangpeng (ORCID 0009-0005-9991-4164)
   - **Description:** 见 README.md 摘要
   - **License:** MIT
   - **Related identifier:** GitHub release URL（Relation: is supplement to）
4. **Publish** → 获得 DOI，例如 `10.5281/zenodo.1234567`
5. 写入 `repository_config.json`：

```json
"zenodo_doi": "10.5281/zenodo.1234567",
"zenodo_url": "https://doi.org/10.5281/zenodo.1234567"
```

6. 重编译 `paper_sensors.pdf`

---

## Step 3 — MDPI Susy 填写

| 字段 | 内容 |
|------|------|
| **Supplementary Materials** | `WeldCascade_Sensors_Supplementary.zip` |
| **Data Availability** | 与稿内 `\dataavailability{{}}` 一致（编译后复制 PDF 段落） |
| **Cover Letter** | `cover_letter_sensors.txt` |

---

## 验证清单

- [ ] GitHub Release `v1.0-sensors` 可匿名访问（无痕窗口打开 Release URL）
- [ ] ZIP 内含 `results/revision_benchmark.json` 与 `priority_experiments.json`
- [ ] `paper_sensors.pdf` Data Availability 含 Release URL（及 Zenodo DOI，若已登记）
- [ ] Cover letter 说明 GDXray 获取障碍与 benchmark 定位

---

## 联系

Yangpeng Zhu — zyp@xsyu.edu.cn
