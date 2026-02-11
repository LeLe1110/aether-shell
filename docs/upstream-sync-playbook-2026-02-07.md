# 上游同步到 develop：操作规范（2026-02-07）

## 1. 目标
- 同步 `upstream/main` 到本地 `develop`，但**按功能点选择**。
- 先判断与本地改动是否冲突；无冲突功能可直接同步，有冲突功能提供解决方案并让你选择。

## 2. 当前仓库关系（摘要）
- `origin`：你的仓库（LeLe1110/aether-bot）
- `upstream`：官方仓库（LeLe1110/aether-bot）
- `main`：稳定/发布基线
- `develop`：日常集成分支
- 公共基点：`16f6fdf`（`develop` 与 `upstream/main` 的 merge-base）

## 3. 本地改动概览（基于 merge-base）
本地（`develop`）自 `16f6fdf` 起的改动文件：
- `.gitignore`
- `docs/session-summary-2026-02-07.md`
- `nanobot/agent/__init__.py`
- `nanobot/agent/context.py`
- `nanobot/agent/context_manager.py`
- `nanobot/agent/loop.py`
- `nanobot/agent/tools/message.py`
- `nanobot/bus/events.py`
- `nanobot/bus/queue.py`
- `nanobot/channels/base.py`
- `nanobot/channels/feishu.py`
- `nanobot/channels/manager.py`
- `nanobot/cli/commands.py`
- `nanobot/config/loader.py`
- `nanobot/config/schema.py`
- `nanobot/providers/base.py`
- `nanobot/providers/litellm_provider.py`
- `nanobot/session/manager.py`
- `scripts/restart-nanobot.sh`

本地核心功能点（与你本次需求相关）：
- 会话绑定 + 上下文压缩（ContextManager）
- LLM session_state 支持（Responses API）
- CLI/飞书显示上下文状态
- Feishu 输出附加“会话模式/压缩/来源”等信息
- `.DS_Store` 忽略规则

## 4. 上游新增功能（按功能点分组）
> 以下是 **upstream/main 相对当前 develop** 的新内容。

### A. Feishu Markdown 渲染增强
- 代表提交：`2ca15f2`
- 文件：`nanobot/channels/feishu.py`
- **冲突：是**（本地也改了 feishu 输出逻辑）
- 解决方案：
  - 先合入上游渲染增强
  - 再保留本地 `_append_context_status` 与“会话状态”输出

### B. AiHubMix Provider 支持 + Provider 匹配重构
- 代表提交：`572eab8`
- 文件：`nanobot/providers/litellm_provider.py`、`nanobot/config/schema.py`、`nanobot/cli/commands.py`、`nanobot/agent/loop.py`、`README.md`
- **冲突：是**（本地改过 provider、config、loop、CLI）
- 解决方案：
  - 合入上游 provider 逻辑后
  - 重新确认：`session_state` 支持、usage 计算、ContextManager 注入点仍保留

### C. DashScope Provider 支持
- 代表提交：`18ec651` / `8499dbf`
- 文件：`nanobot/config/schema.py`、`nanobot/providers/litellm_provider.py`
- **冲突：是**（本地改过 provider/config）
- 解决方案：
  - 保留本地 session_state / usage 处理
  - 兼容新增 DashScope 入口

### D. Zhipu / vLLM API Key 修复
- 代表提交：`a0280a1`、`8cde0b3`（上游还有合并提交）
- 文件：`nanobot/providers/litellm_provider.py`、`nanobot/cli/commands.py`
- **冲突：是**（本地改过 provider/CLI）
- 解决方案：
  - 合入 API key 修复
  - 确保本地 session_state、usage 逻辑不被覆盖

### E. Moonshot Provider 支持
- 代表提交：`e680b73` / `9a8e9bf`
- 文件：`nanobot/config/schema.py`、`nanobot/providers/litellm_provider.py`
- **冲突：是**

### F. 安全加固（workspace 限制、SECURITY.md）
- 代表提交：`5f5536c` / `c5191ee` / `943579b` / `8b4e0a8`
- 文件：`nanobot/agent/tools/filesystem.py`、`nanobot/agent/loop.py`、`nanobot/agent/subagent.py`、`nanobot/channels/base.py`、`nanobot/config/schema.py`、`nanobot/config/loader.py`、`SECURITY.md`、`bridge/package.json` 等
- **冲突：是**（本地改了 loop/channels/config）
- 解决方案：
  - 先合入安全策略，再恢复本地 ContextManager 注入点
  - 注意 `restrictToWorkspace` 配置的变更路径

### G. Discord 通道支持
- 代表提交：`ba6c4b7` / `bab464d` / `be0cbb7`
- 文件：`nanobot/channels/discord.py`、`nanobot/channels/manager.py`、`nanobot/config/schema.py`、`nanobot/cli/commands.py`、`README.md`
- **冲突：是**（本地改了 channels/manager/config/CLI）
- 解决方案：
  - 合入新增通道后，保留 Feishu 的 showContext 逻辑

### H. 文档与脚本更新（无代码冲突）
- 代表提交：`625fc60`、`4617043`、`6bf09e0`、`9d5b227`、`9ac3944`、`cb800e8`、`dcae2c2`、`77d4892` 等
- 文件：`README.md`、`core_agent_lines.sh`、`tests/test_docker.sh` 等
- **冲突：否**（本地未改这些文件）
- 可直接 cherry-pick

### I. .gitignore 更新
- 代表提交：`8a23d54`
- 文件：`.gitignore`
- **冲突：是**（本地加入了 `.DS_Store` 忽略）
- 解决方案：
  - 直接人工合并两份规则

## 5. 选择式同步操作规范（推荐流程）

### Step 0：准备工作
```bash
git status -sb
```
确保当前工作区干净（已提交）。

### Step 1：创建同步分支
```bash
git checkout develop
git checkout -b sync/upstream-2026-02-07
```

### Step 2：查看上游差异清单
```bash
git fetch upstream
git log --oneline develop..upstream/main
```

### Step 3：按“功能点选择同步”
- **无冲突功能**：直接 `git cherry-pick` 对应提交
- **有冲突功能**：按解决方案处理后再选择是否同步

## 6. 功能选择清单（请勾选）

### 无冲突功能（建议直接同步）
- [ ] 文档/脚本更新（README / core_agent_lines.sh / tests/test_docker.sh 等）

### 有冲突功能（需手动合并）
- [ ] Feishu Markdown 渲染增强
- [ ] AiHubMix Provider 支持 + Provider 匹配重构
- [ ] DashScope Provider 支持
- [ ] Zhipu / vLLM API Key 修复
- [ ] Moonshot Provider 支持
- [ ] 安全加固（restrictToWorkspace / SECURITY.md）
- [ ] Discord 通道支持
- [ ] .gitignore 更新（保留 .DS_Store 忽略）

## 7. 冲突处理建议（要点清单）

### Feishu 渲染增强
- 合并上游 `nanobot/channels/feishu.py`
- 保留本地 `_append_context_status()` 及上下文状态输出

### Provider 相关（AiHubMix / DashScope / Moonshot / API Key 修复）
- 合并 `nanobot/providers/litellm_provider.py` / `nanobot/config/schema.py` / `nanobot/cli/commands.py`
- **保留本地**：`session_state`、`response_id`/`usage` 处理、ContextManager 注入逻辑

### 安全加固
- 合并 `restrictToWorkspace` 相关改动
- **保留本地**：ContextManager 与会话绑定流程

### Discord 支持
- 合并 `nanobot/channels/discord.py`
- 更新 `channels/manager.py`、`config/schema.py`、`cli/commands.py`
- 保留 Feishu showContext 相关字段

### .gitignore
- 合并上游规则 + 本地 `.DS_Store` 忽略

## 8. 输出/验收
- 同步完成后，执行：
```bash
git status -sb
git log --oneline -n 5
```
- 如需，运行自测或手动验证通道/Provider。

---

> 备注：本规范允许逐功能同步，避免一次性合并造成大范围冲突。
