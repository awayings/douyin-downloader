# 规范 Kimi 多模态视频转录 Provider 设计

## 背景

当前 `transcript` 模块仅支持 OpenAI Whisper 风格 API（`POST /v1/audio/transcriptions`，`multipart/form-data` 上传视频文件，返回 `{"text": "..."}`）。Kimi（Moonshot）的视觉模型支持直接输入 MP4 视频，但走的是 `/v1/chat/completions` JSON 接口，请求体和响应格式与 Whisper 完全不同。

目标：将 Kimi 集成为一等转录 Provider，与现有 OpenAI Whisper 共存，通过 `api_url` 自动推断使用哪套协议。

## 配置层变更

### config/default_config.py

在 `transcript` 字典中新增 `prompt` 字段：

```python
"transcript": {
    "enabled": False,
    "model": "gpt-4o-mini-transcribe",
    "output_dir": "",
    "response_formats": ["txt", "json"],
    "api_url": "https://api.openai.com/v1/audio/transcriptions",
    "api_key_env": "OPENAI_API_KEY",
    "api_key": "",
    "prompt": "请将这段视频的语音内容转录成文字。",
},
```

### config.yml / config.example.yml

同步追加一行：

```yaml
transcript:
  # ... 现有字段 ...
  prompt: "请将这段视频的语音内容转录成文字。"
```

> `prompt` 仅对 Kimi Provider 生效（Whisper 无 prompt 概念）。用户可按需自定义提示词。

## Provider 架构

新增目录 `core/transcript_providers/`，采用与 `core/user_modes/` 一致的策略模式：

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseTranscriptProvider` 抽象基类 |
| `openai_provider.py` | OpenAI Whisper 协议实现 |
| `kimi_provider.py` | Kimi Chat Completions 多模态实现 |
| `registry.py` | 根据 `api_url` 自动分发 Provider 实例 |

### BaseTranscriptProvider

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class BaseTranscriptProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        *,
        video_path: Path,
        model: str,
        api_key: str,
        api_url: str,
        prompt: str = "",
    ) -> Dict[str, Any]:
        """
        返回统一结构：
        {
            "text": str,            # 转录后的纯文本
            "raw_response": dict,   # provider 原始响应 JSON
        }
        """
```

### Registry

```python
def resolve_provider(api_url: str) -> BaseTranscriptProvider:
    url = api_url.lower()
    if "moonshot.cn" in url:
        return KimiProvider()
    return OpenAIProvider()  # 默认回退，兼容已有配置
```

## 数据流

```
TranscriptManager.process_video()
  ├── registry.resolve(api_url)  → 返回 BaseTranscriptProvider 实例
  ├── provider.transcribe(
  │       video_path=..., model=..., api_key=..., api_url=..., prompt=...
  │     )
  │     ├── 读取视频文件
  │     ├── 构造请求（Kimi: 先上传文件再引用 ms://file_id / OpenAI: multipart/form-data）
  │     ├── POST 到对应端点
  │     └── 统一解析为 {"text": ..., "raw_response": ...}
  └── TranscriptManager._write_outputs(payload) + _record_job()
```

`TranscriptManager` 改动点：
1. `_call_openai_transcription` 整体移除，替换为 registry + provider 调用。
2. `_cfg()` 中新增读取 `prompt` 的逻辑，透传给 provider。
3. 其余逻辑（`process_video` 的异常处理、`_write_outputs`、`_record_job`）**保持不动**。

## OpenAI Provider

将现有 `core/transcript_manager.py:186-216` 的 `multipart/form-data` 逻辑迁移到 `openai_provider.py`：

- **端点**：`POST {api_url}`（默认 `https://api.openai.com/v1/audio/transcriptions`）
- **请求体**：`aiohttp.FormData`，字段 `model`、`response_format=json`、可选 `language`、`file`
- **响应**：原始 JSON 含 `text` / `duration` / `segments` 等，直接包装为 `{"text": payload["text"], "raw_response": payload}`

## Kimi Provider

基于 [Kimi K2.6 Quickstart](https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart) 实现，采用**先上传后引用**的两步流程（视频文件较大，不支持 base64 内联）：

### Step 1：上传视频文件

- **端点**：`POST {base_url}/files`，其中 `base_url` 从用户配置的 `api_url` 推导。
  - 例：`api_url="https://api.moonshot.cn/v1/chat/completions"` → `files_url="https://api.moonshot.cn/v1/files"`
- **请求体**：`multipart/form-data`，字段 `file`（视频二进制流）、`purpose=video`
- **响应**：`{"id": "file-xxx", ...}`，提取 `id` 作为后续引用标识。

### Step 2：调用 Chat Completions

- **端点**：`POST {api_url}`（用户配置的完整 chat completions URL）
- **鉴权**：`Authorization: Bearer {api_key}` + `Content-Type: application/json`
- **请求体**：
  ```json
  {
    "model": "kimi-k2.6",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "video_url",
            "video_url": {
              "url": "ms://file-xxx"
            }
          },
          {
            "type": "text",
            "text": "请将这段视频的语音内容转录成文字。"
          }
        ]
      }
    ]
  }
  ```
- **响应解析**：从 `choices[0].message.content` 提取纯文本，包装为 `{"text": content, "raw_response": full_json}`
- **约束**：视频格式 mp4/mpeg/mov/avi/x-flv/mpg/webm/wmv/3gpp；推荐分辨率 ≤ 2K

> 上传和 completions 两步都在 `KimiProvider` 内部串行完成，`TranscriptManager` 无感知。上传失败直接抛异常，不进入 completions 步骤。

## 响应标准化

两个 provider 返回**完全相同的结构**，下游 `process_video` 和 `_write_outputs` 无需感知 provider 差异：

```python
{
    "text": "转录后的纯文本",          # .transcript.txt 内容来源
    "raw_response": {原始响应 JSON}   # .transcript.json 内容来源
}
```

- `.transcript.txt` 仍从 `payload.get("text")` 写入，逻辑不变。
- `.transcript.json` 仍写入完整 `raw_response`，保留 provider 原始响应供调试。Kimi 的 `.transcript.json` 里会包含 chat completions 的完整结构（含 `usage` 等），与 Whisper 的 JSON 格式不同，但用途一致。

## 错误处理

Provider 内部抛出具体异常，`TranscriptManager` 现有 `try/except` 在 `process_video(:118-160)` 已能接住：

- **Kimi provider**：
  - HTTP 非 200 时抛出 `RuntimeError(f"Kimi transcription failed: status={...}, body={...}")`
  - 返回非预期结构时抛出 `RuntimeError("Kimi returned invalid payload")`
  - 文件上传失败或读取异常直接上抛
- **OpenAI provider**：保持现有行为（`status != 200` 时抛错）。
- **注册表未知 URL**：默认回退到 `OpenAIProvider`，兼容已有配置（含第三方代理）。

## 测试设计

新增 `tests/test_transcript_providers.py`：

1. `test_registry_detects_kimi` — `api_url="https://api.moonshot.cn/v1/chat/completions"` 返回 `KimiProvider`
2. `test_registry_detects_openai` — `api_url="https://api.openai.com/v1/audio/transcriptions"` 返回 `OpenAIProvider`
3. `test_registry_defaults_to_openai` — 未知 URL 回退到 `OpenAIProvider`
4. `test_kimi_provider_uploads_file` — mock `aiohttp`，验证：
   - 先 POST 到 `{base_url}/files`，multipart 含 `purpose=video`
   - 上传响应解析出 `file_id`
5. `test_kimi_provider_builds_request` — mock `aiohttp`，验证：
   - 第二步 POST 到 `{api_url}`，header 含 `Authorization: Bearer ...`
   - body 为 JSON，含 `messages[0].content` 且 `type="video_url"` + `type="text"`
   - `video_url.url` 为 `ms://{file_id}`
6. `test_kimi_provider_normalizes_response` — 输入 chat completions JSON，验证返回 `{"text": "...", "raw_response": {...}}`
7. `test_openai_provider_normalizes_response` — 输入 Whisper JSON，验证返回结构

所有 provider 单测用 `AsyncMock` mock `aiohttp` session，不请求真实网络。`TranscriptManager` 层已有 `test_transcript_manager.py`（如有），可补充 provider 切换的端到端测试。

## 文件改动清单

| 操作 | 路径 |
|------|------|
| 新增 | `core/transcript_providers/__init__.py` |
| 新增 | `core/transcript_providers/base.py` |
| 新增 | `core/transcript_providers/openai_provider.py` |
| 新增 | `core/transcript_providers/kimi_provider.py` |
| 新增 | `core/transcript_providers/registry.py` |
| 新增 | `tests/test_transcript_providers.py` |
| 修改 | `core/transcript_manager.py` — 移除 `_call_openai_transcription`，接入 registry + provider |
| 修改 | `config/default_config.py` — transcript 追加 `prompt` 字段 |
| 修改 | `config.yml` / `config.example.yml` — 追加 `prompt` 示例 |
