# 电工杯 A 题问题三连续调节实施计划

> **For agentic workers:** 在 `feature/min-cost-network-flow` 隔离工作树内执行。用测试先行；连续变量通过 OR-Tools MPSolver 的 SCIP 后端求解，所有停机/购售电开关逻辑由显式二元变量表示。

**目标：** 在 72 吨/日额定装置下，为五种日产量和 24 个风光场景计算连续功率调度，并与问题二离散调度作同口径比较。

**架构：** 使用小时运行比例连续变量 `f[h]`，约束 `0.1 ≤ f[h] ≤ 1` 且 `sum(f)=目标产量/3`。这里把“下限 10%”解释为每小时装置保持至少 10% 负荷，不设置完全停机状态。为避免低谷购电价低于上网价时同一小时同时买卖电，使用网购方向二元变量和功率上界构成互斥约束。若 SCIP 后端不可用，明确返回求解器不可用，不以离散近似冒充连续最优解。

**技术栈：** OR-Tools `pywraplp` + SCIP、Pydantic、现有能源园区数据、独立逐时 validator。

---

## 文件边界

- 新建 `src/math_modeling_agent/energy_park_continuous.py`：单场景 SCIP 调度、24 场景×五档产量、360 天年度汇总和问题二比较。
- 新建 `src/math_modeling_agent/energy_park_continuous_validator.py`：重算目标产量、逐时连续比例、网购/上网互斥、成本及绿电指标。
- 新建 `tests/test_energy_park_continuous.py`：连续比例边界、目标吨数、买卖互斥、场景及 Q2 对比。
- 修改 `src/math_modeling_agent/cli.py`、`tests/test_cli.py`：增加 `--energy-park-q3`。
- 修改 `README.md`：记录 10% 最低功率口径、SCIP 和 Q2 对比输出。

## Task 1：单场景连续调度

- [ ] **步骤 1：写失败测试**。构造 24 小时数据和日产量 36 吨，断言每小时运行比例位于 `[0.1, 1]`、比例总和为 12、产量为 36 吨、购电与上网不会在同一小时同时为正，solver 状态为 `OPTIMAL`。
- [ ] **步骤 2：运行测试并确认连续调度函数不存在**。

```powershell
& D:\Agent\.venv\Scripts\python.exe -m pytest tests\test_energy_park_continuous.py::test_continuous_schedule_respects_minimum_fraction_and_target -q
```

- [ ] **步骤 3：实现 SCIP 模型**。每小时设 `f[h]∈[0.1,1]`、`buy[h]≥0`、`sell[h]≥0` 和方向变量 `grid_import[h]∈{0,1}`；写明功率平衡，约束 `buy[h]≤M_buy·grid_import[h]`、`sell[h]≤M_sell·(1-grid_import[h])`。固定日电量目标写成 `sum(f)=target/(3 t/h)`。目标函数包括分时购电、上网收入及设备运维成本；风光 LCOE 和额定资本摊销按成本报告附加。
- [ ] **步骤 4：实现连续计划 validator**。独立检查 24 个 `f` 的边界、总产量、逐时能量平衡、网购/上网互斥、成本和政策比例；`SCIP` 不存在时返回明确失败状态。
- [ ] **步骤 5：运行该文件所有测试并检查 SCIP 可用性错误路径**。

## Task 2：批量场景和同口径对比

- [ ] **步骤 1：写批量测试**，要求五个产量×24 组合各有一个 `OPTIMAL` 且验证通过的结果，年度权重合计 360 天。
- [ ] **步骤 2：运行测试并确认批量 runner 缺失**。
- [ ] **步骤 3：实现典型日、24 场景调度及年度成本/绿电分类汇总**。允许把问题二结果作为输入；未提供时调用问题二 runner。以同一产量、同一场景、同一成本口径比较离散与连续的吨氨成本、网购/上网和两套绿电达标天数。
- [ ] **步骤 4：运行批量测试并核对 5×24 的组合唯一性**。

## Task 3：CLI 和验收

- [ ] **步骤 1：写 `--energy-park-q3` CLI 测试**，检查目标产量、24 场景结果和验证状态。
- [ ] **步骤 2：运行并确认当前 CLI 不识别该参数**。
- [ ] **步骤 3：接入 CLI，失败时输出明确错误且返回非零退出码；README 记录 SCIP 后端及 10% 最低负荷假设。**
- [ ] **步骤 4：运行 `python -m pytest -p no:cacheprovider -q`，再用 A 题真实附件运行 `python -m math_modeling_agent.cli --energy-park-q3 "D:\Agent\例题\电工杯A"`。**
- [ ] **步骤 5：提交问题三连续调度改动。**
