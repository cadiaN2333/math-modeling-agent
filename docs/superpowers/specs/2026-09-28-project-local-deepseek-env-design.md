# 项目级 DeepSeek 密钥配置设计

## 目标

让数学建模 Agent 从项目根目录的 `.env` 自动读取 `DEEPSEEK_API_KEY`，用户无需每次在 PowerShell 中重新设置环境变量。

## 当前情况

- `analysis_agent.py` 目前直接调用 `os.getenv("DEEPSEEK_API_KEY")`。
- 项目根目录还没有 `.env` 或 `.gitignore`。
- DeepSeek/OpenAI 兼容客户端只在分析请求运行时创建。

## 设计

1. 在 `agent` 可选依赖中加入 `python-dotenv`，不增加纯本地排班求解器的必需依赖。
2. 调用远程模型前，从包文件路径推导项目根目录，并加载根目录 `.env`；使用 `override=False`，避免覆盖用户当前进程中已经明确设置的环境变量。
3. 增加 `.env.example`，仅提供变量名和占位说明，不包含任何真实密钥。
4. 增加 `.gitignore` 并忽略项目根目录 `.env`，保护本地密钥不进入版本控制。
5. 如果安装了项目 `agent` 可选依赖，运行 `python -m math_modeling_agent.cli --request ...` 即会自动加载配置；没有密钥时仍给出明确错误提示。

## 验证

- 单元测试验证 `.env` 加载路径固定指向项目根目录，且不覆盖已有环境变量。
- 单元测试验证 `.env` 被 Git 忽略、`.env.example` 不含密钥值。
- 完整测试套件通过；不在测试中调用真实 DeepSeek API，也不读取用户 `.env` 文件内容。

## 安全边界

真实密钥由用户在本地创建的 `.env` 文件中填写。代码、测试和示例文件都不读取、打印或保存真实密钥。若密钥曾被提交到版本控制，应在 DeepSeek 控制台撤销并重新生成。
