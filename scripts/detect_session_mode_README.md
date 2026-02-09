# Session Mode 检测脚本使用说明

## 概述

`scripts/detect_session_mode.sh` 用于检测 API 端点支持哪种会话模式，帮助你正确配置 nanobot 的 `session_mode` 参数。

## 背景

nanobot 通过 OpenAI Responses API 的 `previous_response_id` 实现服务端会话延续（native 模式）。进入 native 模式后，nanobot 不再发送系统规范（AGENTS.md、SOUL.md 等），依赖服务端记住上下文。

但很多第三方中转站（Sub2API）虽然支持 `/v1/responses` 的请求格式，却**不支持 `previous_response_id`**。如果 nanobot 错误进入 native 模式，会导致：

- 系统规范丢失（bot 失去人设、语言设定）
- bot 回退到 API 默认的 system prompt（如自称 "OpenCode"）
- 用户看到不相关的���文回复

**本脚本在配置新 API 端点前运行一次，即可确定正确的 `session_mode` 配置。**

## 依赖

- Python 3
- httpx（项目 `.venv` 已包含）

## 用法

```bash
# 基本用法
PYTHON=.venv/bin/python3 \
API_KEY="sk-你的key" \
BASE_URL="https://你的api地址" \
bash scripts/detect_session_mode.sh

# 指定模型
PYTHON=.venv/bin/python3 \
API_KEY="sk-xxx" \
BASE_URL="https://example.com/v1" \
MODEL="gpt-5.2" \
bash scripts/detect_session_mode.sh

# 通过代理
PYTHON=.venv/bin/python3 \
API_KEY="sk-xxx" \
BASE_URL="https://example.com" \
PROXY="http://127.0.0.1:7897" \
bash scripts/detect_session_mode.sh
```

## 参数

| 环境变量 | 必填 | 默认值 | 说明 |
|---------|------|--------|------|
| `API_KEY` | 是 | - | API 密钥 |
| `BASE_URL` | 是 | - | API 地址，如 `https://example.com` 或 `https://example.com/v1` |
| `MODEL` | 否 | `gpt-4o` | 模型名称 |
| `PROXY` | 否 | 无 | HTTP 代理地址 |
| `TIMEOUT` | 否 | `60` | 请求超时秒数 |
| `PYTHON` | 否 | `python3` | Python 解释器路径，建议用 `.venv/bin/python3` |

## 检测结果

| 结果 | 含义 | nanobot 配置 |
|------|------|-------------|
| **native** | 支持 `previous_response_id`，服务端保持完整会话上下文 | `"api_type": "openai-responses"`, `"session_mode": "native"` |
| **stateless** | 支持 `/v1/responses` 请求格式，但不支持 `previous_response_id` | `"api_type": "openai-responses"`, `"session_mode": "stateless"` |
| **completions** | 只支持 `/v1/chat/completions`，不支持 Responses API | 不要设 `api_type`，也不需要设 `session_mode` |
| **unavailable** | API 不可用（502、超时等） | 检查网络、API key、源站状态 |

## 检测流程

脚本依次执行 6 个测试：

1. **Test 1 — 连通性** (`GET /v1/models`)：检查 API 是否可达，列出可用模型
2. **Test 2 — Chat Completions** (`POST /v1/chat/completions`)：检查是��支持旧版 API
3. **Test 3 — Responses API** (`POST /v1/responses`)：发送简单请求，检查是否支持 Responses API 格式
4. **Test 4 — System prompt + store** ：发送带系统提示的请求（`store: true`），检查系统提示和存储功能
5. **Test 5 — previous_response_id（关键测试）**：仅通过 `previous_response_id` 引用上一轮，**不发送系统提示**，检查 bot 是否仍记得系统提示内容
6. **Test 6 — 三轮链式请求** ：验证多轮会话记忆（仅在 Test 5 通过时执行）

## nanobot 配置示例

根据检测结果，在 `~/.nanobot/config.json` 中配置对应的 provider：

### native 模式（API 完全支持 `previous_response_id`）

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-xxx",
      "api_base": "https://api.openai.com/v1",
      "api_type": "openai-responses",
      "session_mode": "native"
    }
  }
}
```

### stateless 模式（Sub2API 中转站常见情况）

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-xxx",
      "api_base": "https://your-relay.example.com/v1",
      "api_type": "openai-responses",
      "session_mode": "stateless"
    }
  }
}
```

### 不设置（auto 模式）

如果不设置 `session_mode`，nanobot 会自动检测：
- 首次使用 `previous_response_id` 时，如果 API 返回 400 `"Unsupported parameter: previous_response_id"`，nanobot 会自动降级为 stateless 模式
- 降级后在当前进程生命周期内持续生效，无需重启

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-xxx",
      "api_base": "https://your-relay.example.com/v1",
      "api_type": "openai-responses"
    }
  }
}
```

> **建议**：虽然 auto 模式能工作，但**首次请求会浪费一次 API 调用**（发送 → 收到 400 → 重试）。如果已经用脚本检测过，直接配置 `session_mode` 更高效。

## 已测试的端点

| 端点 | 检测结果 | 说明 |
|------|---------|------|
| `https://right.codes/codex/v1` | **stateless** | 支持 Responses API 格式，`previous_response_id` 返回 400，`store` 强制 false |
| `https://gmn.chuangzuoli.com` | **unavailable** | 间歇性 502（源站不稳定），chat/completions 返回 404 |
| `https://api.openai.com/v1` | **native**（预期） | OpenAI 官方 API，完全支持 |

## 常见问题

### Q: 脚本报 `httpx not installed`
A: 使用项目 venv：`PYTHON=.venv/bin/python3 API_KEY=... BASE_URL=... bash scripts/detect_session_mode.sh`

### Q: Test 3 超时
A: 增加超时：`TIMEOUT=120`，或检查是否需要代理：`PROXY="http://127.0.0.1:7897"`

### Q: 检测为 stateless 但我知道 API 支持 native
A: 可能是 `store` 被强制为 false 导致服务端没有保存上下文。检查 Test 4 的 Store 字段。

### Q: 不设 session_mode 会怎样？
A: nanobot 会尝试 native 模式，如果 API 返回 400 会自动降级。但首次请求会多花一次 API 调用，且如果 API 静默忽略 `previous_response_id`（返回 200 但不保留上下文），nanobot 无法检测到，会导致系统规范丢失。**建议用本脚本检测后明确配置。**
