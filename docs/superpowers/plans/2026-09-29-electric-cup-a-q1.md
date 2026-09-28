# 电工杯 A 题问题一基准计算实施计划

> **For agentic workers:** 本计划在 `feature/min-cost-network-flow` 隔离工作树内执行。逐项使用测试先行：先写一个行为测试、运行并确认按预期失败、再写最少生产代码、复跑测试。代码注释和文档均用中文。

**目标：** 从题目附件读取典型日数据，计算逐时功率、电量、绿电指标的两种自用口径、吨氨成本组成，并由独立校验器复核。

**架构：** 新增能源园区输入模型、只读 Excel 加载器、问题一确定性功率平衡计算器和独立结果校验器。CLI 增加显式的电工杯问题一入口；输出包括建模思路、来源、审阅事项、逐时结果及成本口径。现有排班、LP 和网络流模块不改。

**技术栈：** Python 3.11、Pydantic、openpyxl 只读加载 XLSX、pytest。求解器不参与问题一，因为该题小问给定装置连续满负荷运行，没有决策变量。

---

## 文件边界

- 新建 `src/math_modeling_agent/energy_park_models.py`：小时曲线、设备参数、成本组件及结果的数据模型和校验。
- 新建 `src/math_modeling_agent/energy_park_io.py`：按附件名称只读加载 1—8，校验表结构、小时标签、数值范围、场景数量和成本单位。
- 新建 `src/math_modeling_agent/energy_park.py`：问题一逐时功率平衡、日电量和成本计算。
- 新建 `src/math_modeling_agent/energy_park_validator.py`：独立重算逐时平衡、电量、指标和成本。
- 修改 `src/math_modeling_agent/cli.py`：增加 `--energy-park-q1 目录`，输出 UTF-8 JSON，并按校验状态返回退出码。
- 修改 `pyproject.toml`：新增可选依赖 `energy = ["openpyxl>=3.1"]`。
- 新建 `tests/test_energy_park_models.py`、`tests/test_energy_park_io.py`、`tests/test_energy_park.py`；扩展 `tests/test_cli.py`。
- 修改 `README.md`：记录安装命令、运行示例和“目前实现问题一基准”的范围。

## Task 1：固定 24 时段的数据契约

**文件：** 新建 `tests/test_energy_park_models.py`、`src/math_modeling_agent/energy_park_models.py`。

- [ ] **步骤 1：写失败测试**

```python
import pytest
from pydantic import ValidationError

from math_modeling_agent.energy_park_models import HourlyProfile


def test_hourly_profile_rejects_23_values() -> None:
    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    with pytest.raises(ValidationError):
        HourlyProfile(periods=periods, values=[0.5] * 23)
```

- [ ] **步骤 2：运行并确认因模型不存在而失败**

```powershell
Set-Location D:\Agent\.worktrees\min-cost-network-flow
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park_models.py::test_hourly_profile_rejects_23_values -q
```

- [ ] **步骤 3：实现最小数据模型**

`HourlyProfile` 固定 24 个不重复的时段标签和 24 个有限标幺值，标幺值范围为 `[0, 1]`。`EnergyParkDataset` 保存常规负荷、典型日风光、6 条风电场景、4 条光伏场景、设备和电价参数。所有曲线必须使用完全相同的小时标签，风电和光伏场景数必须分别为 6 和 4。校验器还需核对额定功率、效率、单位耗电和产氢/耗氢数据是否一致。

- [ ] **步骤 4：复跑并补边界测试**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park_models.py -q
```

测试 24 个值通过；23/25 个值、重复时段、NaN、负标幺值、超过 1 的值及场景数量错误均被拒绝。

## Task 2：XLSX 附件读取

**文件：** 新建 `tests/test_energy_park_io.py`、`src/math_modeling_agent/energy_park_io.py`；修改 `pyproject.toml`。

- [ ] **步骤 1：写表格矩阵解析测试**

```python
from math_modeling_agent.energy_park_io import parse_hourly_table


def test_parse_hourly_table_preserves_period_order() -> None:
    periods = [f"{hour}:00-{hour + 1}:00" for hour in range(24)]
    rows = [("时段", "标幺值"), *[(period, 0.5) for period in periods]]

    result = parse_hourly_table(rows, source_name="附件1")

    assert result.periods == periods
    assert result.values == [0.5] * 24
```

再加缺失单元格和错误列标题测试，要求错误包含来源名及所在行/列。

- [ ] **步骤 2：运行并确认因解析函数不存在而失败**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park_io.py::test_parse_hourly_table_preserves_period_order -q
```

- [ ] **步骤 3：实现只读加载器**

实现 `load_energy_park_directory(directory: Path) -> EnergyParkDataset`。使用 `openpyxl.load_workbook(path, read_only=True, data_only=True)`，逐表解析附件 1—8，不保存或覆盖源文件。遇到缺文件、错位小时、空值、非数值、单位或行列标题不符时抛出带附件名和位置的 `ValueError`，并在数据模型中保留实际使用的来源文件名。缺少 openpyxl 时提示安装 `python -m pip install -e ".[dev,energy]"`。

- [ ] **步骤 4：跑单元测试和原始附件集成检查**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park_io.py -q
& D:\Agent\.venv\Scripts\python.exe -c "from pathlib import Path; from math_modeling_agent.energy_park_io import load_energy_park_directory; x=load_energy_park_directory(Path(r'D:\Agent\例题\电工杯A')); print(len(x.periods), len(x.wind_scenarios), len(x.pv_scenarios))"
```

原始附件集成检查应输出 `24 6 4`。

## Task 3：问题一功率平衡和指标

**文件：** 新建 `tests/test_energy_park.py`、`src/math_modeling_agent/energy_park.py`。

- [ ] **步骤 1：写确定性计算测试**

构造 24 小时合成曲线：常规负荷标幺值均为 `1/6`；前 12 小时风光为零，后 12 小时风电标幺值 `0.5`、光伏标幺值 `0.25`；装置功率为题面值 `10、10、0.75 MW`。每小时负荷为 `21.75 MW`。预期总用电 `522 MWh`、新能源发电 `432 MWh`、购电 `261 MWh`、上网 `171 MWh`，逐小时平衡残差为零。检验题面字面自用率和物理自用率被分别输出。

- [ ] **步骤 2：运行并确认因计算函数不存在而失败**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park.py::test_typical_day_balances_buy_and_export_power -q
```

- [ ] **步骤 3：实现基准计算器**

实现 `compute_typical_day(data: EnergyParkDataset) -> TypicalDayResult`。对每小时计算常规负荷、设备负荷、风光发电、购电和上网。购电与上网分别是净缺口的正部和净富余的正部。以 1 小时间隔将 MW 求和为 MWh。逐项返回题面自用率 `(总用电-上网-网购)/新能源发电`、物理自用率 `(新能源发电-上网)/新能源发电`、总用电绿电比例及上网比例，并分别按题面阈值判断。

- [ ] **步骤 4：复跑并检查边界**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park.py -q
```

测试汇总值误差小于 `1e-8`；净功率为零时购电和上网都为零；新能源发电量为零时比例返回明确的不可计算状态，不返回 NaN 或伪造零值。

## Task 4：成本明细与独立校验

**文件：** 新建 `src/math_modeling_agent/energy_park_validator.py`；扩展 `tests/test_energy_park.py`。

- [ ] **步骤 1：写篡改结果测试**

计算合法结果后，把某小时上网功率增加 `0.1 MW`，断言 validator 报告无效；再把汇总指标改成错误值，断言 validator 从原始输入重算并发现差异。

- [ ] **步骤 2：运行并确认因 validator 不存在而失败**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park.py::test_validator_detects_tampered_power_balance -q
```

- [ ] **步骤 3：实现独立核验和成本组成**

`validate_typical_day(data, result)` 不使用 `result` 中已汇总的指标作为核验输入，而是重新计算电力平衡、电量、指标和成本。成本逐项列出风光度电成本、分时购电成本、上网收入、电解槽运维费、合成氨运维费及固定资产摊销。

`60000 元/kgH₂` 投资成本只在明确标记的“按铭牌氢处理能力、直线折旧”情形下纳入资本成本；同时报告未计该摊销的运行成本，并列出附件未提供的电解槽资本成本。不能把单一口径称为唯一吨氨成本。

- [ ] **步骤 4：复跑并验证篡改检测**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park.py -q
```

正常解校验通过；任一小时功率、电量、绿电指标或成本组成被篡改时校验失败。

## Task 5：CLI、文档和真实附件验收

**文件：** 修改 `src/math_modeling_agent/cli.py`、`tests/test_cli.py`、`README.md`。

- [ ] **步骤 1：写 CLI 行为测试**

增加 `--energy-park-q1 <附件目录>`。用猴子补丁注入合成输入目录，检验输出 JSON 包含建模思路、数据来源、逐时曲线、双口径绿电指标、成本组成和 validator 报告。附件缺失返回退出码 `2`，校验失败返回 `1`。

- [ ] **步骤 2：运行并观察预期失败**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_cli_energy_park_q1_outputs_balanced_results -q
```

预期：当前 CLI 不识别 `--energy-park-q1`。

- [ ] **步骤 3：实现独立 CLI 分支并更新 README**

新增显式参数入口，调用 Excel 加载器、基准计算器和 validator。其它已有参数行为保持不变。README 增加安装和运行说明：

```powershell
python -m pip install -e ".[dev,energy]"
python -m math_modeling_agent.cli --energy-park-q1 "D:\Agent\例题\电工杯A"
```

- [ ] **步骤 4：全量测试并用原始附件复算**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
& D:\Agent\.venv\Scripts\python.exe -m math_modeling_agent.cli --energy-park-q1 "D:\Agent\例题\电工杯A"
```

原始附件预期值：总用电 `558.7200 MWh`、新能源发电 `603.4480 MWh`、购电 `172.0438 MWh`、上网 `216.7718 MWh`；总电量残差不超过 `1e-6 MWh`。题面自用率约 `28.1556%`，物理自用率约 `64.0778%`。成本必须显示口径和摊销假设。

- [ ] **步骤 5：提交本阶段**

```powershell
git add pyproject.toml README.md src/math_modeling_agent/energy_park_models.py src/math_modeling_agent/energy_park_io.py src/math_modeling_agent/energy_park.py src/math_modeling_agent/energy_park_validator.py src/math_modeling_agent/cli.py tests/test_energy_park_models.py tests/test_energy_park_io.py tests/test_energy_park.py tests/test_cli.py
git commit -m "增加电工杯 A 题问题一基准模型"
```

本阶段只完成问题一。后续依次增加问题二 CP-SAT 开停调度、问题三连续功率调度、问题四储能和容量设计、问题五带来源的定性报告；不得据此宣称整道 A 题已经通过验收。

## Task 6：自然语言草稿审阅后再求解

**文件：** 修改 `src/math_modeling_agent/cli.py`、`tests/test_cli.py`、`README.md`。

- [ ] **步骤 1：将 ready 请求测试改为确认求解器未调用**

保留现有 ready 排班分析夹具，替换原来直接求解的断言：猴子补丁把 `run_modeling` 替换为立即触发 `pytest.fail()` 的函数；调用 `main(["--request", ...])` 后断言有结构化草稿、状态为 `ready`，且 JSON 不含 `modeling_run`。

- [ ] **步骤 2：运行并确认旧自动求解行为使测试失败**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_cli_ready_request_returns_draft_without_solving -q
```

预期：旧代码调用 `run_modeling`，触发测试中的 `pytest.fail()`。

- [ ] **步骤 3：增加显式确认的草稿文件入口**

将 `--solve-draft <JSON文件>` 加入 CLI 互斥输入组。`--request` 只输出分析和方法建议。`--solve-draft` 读取完整 `analysis` 对象、调用 `ProblemAnalysis.model_validate()` 重新校验，然后按已确认的 `problem_family` 调用对应领域适配器。文件格式无效、状态不是 `ready`、或领域没有实现适配器时返回退出码 `2`；求解未通过 validator 时返回 `1`。

- [ ] **步骤 4：测试显式确认路径**

在 `tests/test_cli.py` 中构造 ready 排班分析，将 `{"analysis": analysis.model_dump(mode="json"), "method_recommendations": {}}` 写入 `tmp_path / "draft.json"`，调用 `main(["--solve-draft", str(path)])`，断言求解状态可行且 validator 通过。另测畸形 JSON 返回 `2`，非 ready 草稿不会调用求解器。

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_cli_ready_request_returns_draft_without_solving tests\test_cli.py::test_cli_solve_confirmed_draft_runs_local_solver -q
```

- [ ] **步骤 5：更新 README 的真实流程**

说明 `--request` 只产出待审阅模型；保存输出 JSON、在 PyCharm 中检查变量/目标/约束/来源，再通过 `--solve-draft` 显式求解。说明 `--energy-park-q1` 是题一计算入口，并保留安装 `.[dev,agent,energy]` 的 PowerShell 命令。

- [ ] **步骤 6：回归测试和提交**

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
git add README.md src/math_modeling_agent/cli.py tests/test_cli.py
git commit -m "要求确认自然语言模型后再求解"
```
