# 项目级 DeepSeek `.env` 自动加载实施计划

> **执行说明：** 按任务逐项实施，复选框用于跟踪进度；代码变更先写失败测试，再完成实现。

**目标：** 让项目 CLI 从 `D:\Agent\.env` 自动读取 DeepSeek 配置，免去每次在 PowerShell 中重新设置。

**架构：** 在默认模型客户端创建前，通过 `python-dotenv` 从 `analysis_agent.py` 路径推导项目根目录并加载 `.env`，不覆盖已存在的进程变量。真实密钥放在本地忽略文件，代码仓库仅提供空配置模板和中文使用说明。

**技术栈：** Python、python-dotenv、pytest、PowerShell。

---

## 文件职责

- `pyproject.toml`：为 `agent` 可选依赖加入 `python-dotenv`。
- `src/math_modeling_agent/analysis_agent.py`：添加项目环境加载函数，并在默认客户端路径调用。
- `tests/test_analysis_agent.py`：验证加载路径、不覆盖进程变量以及默认分析入口自动加载。
- `tests/test_project_config.py`：验证示例项为空、Git 忽略规则精确。
- `.gitignore`：忽略真实 `.env`。
- `.env.example`：提供空 API Key 和非敏感默认模型、地址。
- `docs/本地运行与密钥配置.md`：说明在 PyCharm 中复制模板、填密钥、安装依赖和运行 CLI。
- `.env`：若当前不存在则创建空白本地模板；绝不覆盖或读取已有文件。

## 任务一：TDD 增加环境加载器单测

- [ ] 在 `tests/test_analysis_agent.py` 添加以下测试：注入假 `dotenv` 模块，确认加载目标为项目根目录 `.env` 且不覆盖当前环境：

```python
def test_load_project_environment_uses_root_env_without_override(monkeypatch) -> None:
    import sys
    from pathlib import Path
    from types import ModuleType

    from math_modeling_agent.analysis_agent import load_project_environment

    calls = []
    fake_dotenv = ModuleType("dotenv")

    def fake_load_dotenv(*, dotenv_path, override):
        calls.append((Path(dotenv_path), override))

    fake_dotenv.load_dotenv = fake_load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    load_project_environment()

    expected_env = Path(__file__).resolve().parents[1] / ".env"
    assert calls == [(expected_env, False)]
```
- [ ] 先运行以下 PowerShell 命令，预期因 `load_project_environment` 尚不存在而失败：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_analysis_agent.py::test_load_project_environment_uses_root_env_without_override -q
```

## 任务二：实现环境加载器

- [ ] 将可选依赖改为 `agent = ["openai", "python-dotenv>=1.0"]`。
- [ ] 在 `analysis_agent.py` 新增函数：

```python
from pathlib import Path


def load_project_environment() -> None:
    """从项目根目录加载本地环境变量，不覆盖当前进程已设置的值。"""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            '缺少环境变量加载依赖；请运行 python -m pip install -e ".[dev,agent]"'
        ) from exc

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env", override=False)
```

- [ ] 在 `analyze_problem()` 的 `client is None` 分支先调用 `load_project_environment()`，再读取 `DEEPSEEK_API_KEY`；注入测试客户端时不加载磁盘文件。
- [ ] 安装并验证：

```powershell
python -m pip install -e '.[dev,agent]'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_analysis_agent.py -q
```

预期：分析器测试全部通过，不访问真实 API。

## 任务三：验证 CLI 默认客户端会自动加载

- [ ] 在 `tests/test_analysis_agent.py` 添加以下默认客户端集成测试。测试只构造假的 SDK 模块，不会发送网络请求：

```python
def test_analyzer_loads_project_environment_before_default_client(monkeypatch) -> None:
    import sys
    from types import ModuleType

    from math_modeling_agent import analysis_agent
    from math_modeling_agent.analysis_agent import ProblemAnalysis, SchedulingDraft

    expected = ProblemAnalysis(
        status="needs_clarification",
        summary="需要员工信息。",
        known_facts=[],
        missing_information=["员工人数"],
        clarifying_questions=["有多少名员工？"],
        unsupported_reasons=[],
        subtasks=[],
        scheduling_draft=SchedulingDraft(
            employees=[], shifts=[], coverage_requirements=[]
        ),
    )
    calls = []
    monkeypatch.setattr(
        analysis_agent,
        "load_project_environment",
        lambda: calls.append("load_env"),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")

    fake_openai = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            calls.append(("client", api_key))
            self.responses = FakeResponses(expected)

    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = analysis_agent.analyze_problem("需要安排员工班次", client=None)

    assert result == expected
    assert calls == ["load_env", ("client", "test-only-key")]
```
- [ ] 先运行以下命令，预期因加载函数未被调用而失败：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_analysis_agent.py::test_analyzer_loads_project_environment_before_default_client -q
```

- [ ] 在默认客户端路径接入加载函数后重跑该测试，预期通过且无网络请求。

## 任务四：添加本地配置模板、忽略规则和说明

- [ ] 新增 `tests/test_project_config.py`，内容如下：

```python
from pathlib import Path


def test_env_template_is_safe_and_root_env_is_ignored() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ignore_rules = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    template_lines = (project_root / ".env.example").read_text(encoding="utf-8").splitlines()

    assert "/.env" in ignore_rules
    key_line = next(line for line in template_lines if line.startswith("DEEPSEEK_API_KEY="))
    assert key_line == "DEEPSEEK_API_KEY="
    assert "DEEPSEEK_MODEL=deepseek-flash" in template_lines
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com" in template_lines
```
- [ ] 先运行配置测试，预期因配置文件尚不存在而失败：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider .\tests\test_project_config.py -q
```

- [ ] 新增 `.gitignore`：

```gitignore
# 本地 DeepSeek 密钥配置，不要提交
/.env
```

- [ ] 新增 `.env.example`：

```dotenv
# 在本地 .env 文件填写真实密钥，不要提交真实密钥
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

- [ ] 新增 `docs/本地运行与密钥配置.md`，说明将模板复制为 `.env`，只在本机填写 API Key；首次运行 `python -m pip install -e '.[dev,agent]'`，之后在 PyCharm 项目终端直接运行下列真实示例命令，无须再次设置环境变量：

```powershell
python -m math_modeling_agent.cli --request '安排员工班次：三名员工中林晓有急救技能，两个八小时班次每班至少一人，第一班需要一名急救员。'
```
- [ ] 若 `.env` 不存在，创建空白本地模板；若已存在则保持不动，不读取内容。
- [ ] 重跑配置测试，预期通过且不读取真实 `.env`。

## 任务五：完整回归测试

- [ ] 运行完整测试套件：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

预期：全部测试通过，无 DeepSeek 网络调用。
- [ ] 仅检查密钥是否可见，不打印密钥：

```powershell
python -c "import os; print('密钥已加载' if os.getenv('DEEPSEEK_API_KEY') else '未加载')"
```

预期：用户填写 `.env` 并重启 PyCharm 终端后显示“密钥已加载”。真实模型请求留给用户自行执行。

## 版本控制说明

当前 `D:\Agent` 没有 `.git` 目录，因此不执行 Git 提交；验证将通过测试和文件检查完成。
