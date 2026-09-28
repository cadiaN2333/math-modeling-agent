# 电工杯 A 题问题二离散调度实施计划

> **For agentic workers:** 在 `feature/min-cost-network-flow` 隔离工作树内执行。每一行为先写失败测试、观察预期失败、写最少实现、再运行测试。中文注释及文档保持项目约定。

**目标：** 针对 72 吨/日额定电氢氨装置，求解五种日产量在典型日和 24 个风光组合中的最低成本开停时段，并生成逐场景绿电与年度统计。

**架构：** 将问题一计算器扩展为给定逐时运行比例和装置规模的统一成本核算函数。CP-SAT 只选择每小时开机或停机；小时买电、上网及其成本由确定性功率平衡计算，不引入可同时买卖的决策变量。绿电指标按题面阈值作为优化后的评价结果，不作为 Q2 的硬约束。

**技术栈：** 现有 OR-Tools CP-SAT、Pydantic、openpyxl、pytest。功率以 `0.0001 MW` 整数单位表示，成本目标以 `0.00001 元` 缩放成整数，报告值仍由原始浮点数据独立重算。

---

## 文件边界

- 修改 `src/math_modeling_agent/energy_park.py`、`energy_park_validator.py`：支持场景曲线、容量倍数、逐时运行比例和对应生产量。
- 新建 `src/math_modeling_agent/energy_park_discrete.py`：单场景 CP-SAT 开停优化、24 场景与五种产量批量运行、360 天代表年汇总。
- 新建 `tests/test_energy_park_discrete.py`：开机时段、目标吨数、场景组合、绿电分组及年度权重测试。
- 修改 `tests/test_energy_park.py`：确认问题一 wrapper 等价于 100% 连续开机计划。
- 修改 `src/math_modeling_agent/cli.py`、`tests/test_cli.py`：增加 `--energy-park-q2`。
- 修改 `README.md`：说明问题二的求解范围及绿电指标是事后评价。

## Task 1：泛化逐时成本核算

**文件：** `tests/test_energy_park.py`、`src/math_modeling_agent/energy_park.py`、`src/math_modeling_agent/energy_park_validator.py`。

- [ ] 写测试：`compute_energy_schedule(data, fractions, capacity_scale, wind, pv)` 接受 24 个 `[0,1]` 运行比例；额定设备功率随 `capacity_scale` 同比放大；产氨量等于额定吨/小时 × 倍数 × 各小时比例之和。
- [ ] 先运行测试，确认因函数不存在而失败。
- [ ] 实现统一计划函数；将原 `compute_typical_day(data)` 保留为全小时 `fraction=1, capacity_scale=1` 的兼容包装。
- [ ] 校验器独立重算逐时平衡、产量、电费、绿电比例及成本，读取输出中的倍数和比例作复算输入。
- [ ] 运行 `python -m pytest tests/test_energy_park.py -q`；现有问题一实际附件基准必须不变。

## Task 2：单场景 CP-SAT 开停优化

**文件：** 新建 `tests/test_energy_park_discrete.py`、`src/math_modeling_agent/energy_park_discrete.py`。

- [ ] 写合成场景测试：12 小时没有风光，12 小时有足够光伏；日产量 36 吨要求 12 个开机小时；预期最优选择全部 12 个光伏充足小时，状态为 `OPTIMAL`，产量为 36 吨，独立校验通过。
- [ ] 先运行，确认由于 `solve_discrete_schedule()` 尚不存在而失败。
- [ ] 实现 24 个 BoolVar；产量约束为 `sum(on[h]) == target_tons / (2 × 1.5 t/h)`。每个小时分别用开机/停机功率平衡计算净购电成本，目标系数按 `100000` 缩放为 CP-SAT 整数。固定风光发电成本和日摊销成本在求解后加回报告。
- [ ] 若目标日产量不能由额定产率和整数小时实现，返回清楚的输入错误，不擅自四舍五入。
- [ ] 独立校验开机小时、日产量、小时功率平衡、购售电、成本分项和四个绿电指标。
- [ ] 运行 `python -m pytest tests/test_energy_park_discrete.py -q`；再运行 `tests/test_energy_park.py` 确认 Q1 未回归。

## Task 3：典型日、24 场景和代表年

**文件：** `src/math_modeling_agent/energy_park_discrete.py`、`tests/test_energy_park_discrete.py`。

- [ ] 写测试：生成日产量 `72、63、54、45、36`；结果包含典型日五个调度及 24×5 个场景调度。
- [ ] 写测试：24 个组合按每个风电场景与每个光伏场景的笛卡尔积生成且 ID 唯一。
- [ ] 写测试：每场景 15 天时年度统计是 360 天；按三项绿电指标全满足、部分满足、全不满足分类，分类天数合计 360。
- [ ] 实现批量调度和年度逐场景加权；每个日产量分别给出年度总成本、总产氨量和加权吨氨成本，不把 360 天假称为 365 天。
- [ ] 运行 `python -m pytest tests/test_energy_park_discrete.py -q`。

## Task 4：CLI、文档和回归

**文件：** `src/math_modeling_agent/cli.py`、`tests/test_cli.py`、`README.md`。

- [ ] 写 CLI 测试：`--energy-park-q2 <附件目录>` 返回五种日产量和 24 场景的结果；缺文件返回退出码 `2`，任一 solver/validator 失败不得返回 `0`。
- [ ] 先运行测试，确认当前 CLI 不接受该参数。
- [ ] 实现命令入口及 JSON 汇总；完整逐时结果保留在可复核结构中。
- [ ] README 说明 72 吨/日额定产能、五档产量、24 场景、360 天加权及事后绿电指标评价。
- [ ] PowerShell 运行 `python -m pytest -p no:cacheprovider -q` 和 `python -m math_modeling_agent.cli --energy-park-q2 "D:\Agent\例题\电工杯A"`。
- [ ] 检查 `git diff --check` 后提交，中文提交说明写明问题二离散调度。

## 明示口径

- 装置额定功率按 72 吨/日规模同步放大两倍；开机时三套装置同步满负荷，停机时均为零。
- 每个目标日产量等于 3 吨/小时 × 开机小时数，故对应 24、21、18、15、12 小时。
- 本问按最低成本求排程，再计算政策比例及是否达标；违反指标的可行排程仍照实报告。
- 代表年按题面 24 场景各 15 天计为 360 天；资本成本口径沿用问题一明示假设，报告中单独列出。
- 该阶段只完成问题二，不把问题一和二通过称为 A 题整体验收。
