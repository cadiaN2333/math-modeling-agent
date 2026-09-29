# 运筹优化数学建模 Agent

项目将自然语言问题整理成可审阅的结构化模型，并用 OR-Tools 求解和独立校验。目前实现员工排班、连续/混合整数单目标线性规划、单商品最小费用流，以及电工杯 A 题问题一至三和基于问题二、三结果的问题五分析。

## 环境

- Python 3.11 或更高版本
- OR-Tools
- Pydantic
- pytest（开发和测试）
- openpyxl（读取能源园区 XLSX 附件）

## 安装

在项目根目录 `D:\Agent` 的 PowerShell 终端中运行：

```powershell
python -m pip install -e ".[dev,agent,energy]"
```

## 运行排班样例

```powershell
python -m math_modeling_agent.cli --sample
```

程序会输出 JSON，包括求解状态、员工与班次分配、目标值和独立验证报告。退出码为 `0` 表示已找到并通过验证的排班；非 `0` 表示无可行方案或验证未通过。

## 从 JSON 文件读取问题

```powershell
python -m math_modeling_agent.cli --input .\examples\schedule.json
```

JSON 文件需要包含 `employees`、`shifts` 和 `coverage_requirements` 三个字段。例如：

```json
{
  "employees": [
    {
      "employee_id": "E1",
      "name": "林晓",
      "skills": ["急救"],
      "max_hours": 8
    }
  ],
  "shifts": [
    {
      "shift_id": "S1",
      "duration_hours": 8
    }
  ],
  "coverage_requirements": [
    {
      "shift_id": "S1",
      "minimum_employees": 1,
      "required_skill_counts": {
        "急救": 1
      }
    }
  ]
}
```

`skills` 是字符串数组；`required_skill_counts` 的键是技能名称，值是该班所需的具备该技能的人数。程序会在求解前校验 JSON 和排班字段。

## 电工杯 A 题问题一

安装 `energy` 可选依赖后，提供题目附件目录即可计算典型日 24 小时功率平衡：

```powershell
python -m math_modeling_agent.cli --energy-park-q1 "D:\Agent\例题\电工杯A"
```

结果包括负荷、风光出力、购售电、绿电比例的题面口径和物理口径、成本组成及独立校验报告。第一项自用比例的题面公式会与物理口径分别报告；固定资本摊销假设也会写入结果。成本敏感度按风光度电成本、峰平谷电价、风光上网电价、设备运维率及资本参数分别上调 1% 计算精确边际影响，不假设外部波动区间，并保持当前调度与产量不变，因此不等于重新优化后的影响。当前此入口只实现问题一，不代表 A 题五问均已完成。

## 电工杯 A 题问题二

使用 CP-SAT 为 72 吨/日额定产能的装置安排全开/停产时段，计算五档日产量和 24 个风光组合：

```powershell
python -m math_modeling_agent.cli --energy-park-q2 "D:\Agent\例题\电工杯A"
```

每个风光场景按 15 天加权，因此代表年为 360 天。结果同时按题面第一项公式和物理自用口径统计绿电指标；最低成本排程先按成本求解，指标作为结果评价，不作为硬约束。

## 电工杯 A 题问题三

连续调度使用 OR-Tools SCIP 后端，按每小时 10%—100% 的连续负荷比例满足目标日产量，并排除同一时段同时购电和上网：

```powershell
python -m math_modeling_agent.cli --energy-park-q3 "D:\Agent\例题\电工杯A"
```

当前把 10% 解释为每个小时都需保持的最低运行比例，不包含完全停机。输出给出五种日产量、24 个风光场景的连续调度、360 天汇总，并与问题二相同产量的离散调度比较。

## 电工杯 A 题问题五

根据问题二、三的真实场景结果生成至少三项系统影响和政策建议，并为每项结论附上可核查的官方来源：

```powershell
python -m math_modeling_agent.cli --energy-park-q5 "D:\Agent\例题\电工杯A"
```

该报告不会模拟配电网潮流、电压或备用市场。它会把这些未建模因素列为局限，供论文讨论时区分定量结果与定性依据。

## 运行测试

```powershell
python -m pytest -p no:cacheprovider -q
```

## 连续与混合整数线性规划

线性问题的每个变量都带有 `domain`：`continuous`、`integer` 或 `binary`。全连续模型使用 GLOP；模型只要含整数/二进制变量就使用 SCIP。整数/二进制域会进入独立 validator；用户没有说明变量是否不可分割且该区别会改变方案时，分析阶段应先追问。当前只支持单一线性目标和线性约束，非线性、多目标模型不会伪装成 LP 求解。

可先生成自然语言草稿，检查 `domain` 后再确认求解：

```powershell
python -m math_modeling_agent.cli --request "单一产品x的产量必须是非负整数，目标是最大化产量；容量约束为2x不超过5。"
```

保存并检查 Draft 中变量 `x` 的 `domain` 为 `integer` 后，沿用 `--solve-draft` 入口求解；结果会标明 `solver_name: "SCIP"`，并由 validator 检查整数性、二进制取值、约束及目标值。

## HMML 方法检索

项目使用领域、子领域、方法节点三层结构保存建模知识，并在求解前根据问题描述和目标推荐方法。知识库范围、检索打分和当前支持边界见 docs/HMML设计说明.md。

## 使用 DeepSeek 分析自然语言问题

自然语言分析使用 DeepSeek V4.1 Flash；DeepSeek API 中对应的模型名是 deepseek-flash。模型输出会经过 Pydantic 结构校验，再为每个子任务检索 HMML。API 密钥只从环境变量读取，不要写进源码或提交到仓库。

在项目根目录创建 `.env`，写入自己的模型配置；不要把真实密钥写进源码或提交到 Git：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-flash
```

先分析并生成结构化草稿：

```powershell
python -m math_modeling_agent.cli --request "有三名员工，林晓有急救技能且最多工作八小时；安排两个八小时班次，每班至少一人，第一个班必须有急救技能。" | Set-Content -Encoding utf8 .\draft.json
```

检查 `draft.json` 中的事实、变量、目标和约束，确认它们符合原始问题后，再显式提交求解：

```powershell
python -m math_modeling_agent.cli --solve-draft .\draft.json
```

`--request` 只分析，不启动求解器。`--solve-draft` 会重新校验草稿，再选择已实现的领域适配器并独立检查结果。

## 当前模型范围

- 每个班次有最低人员数要求。
- 班次可以要求一定数量的特定技能员工。
- 每名员工有最大总工时。
- 优化目标是满足约束的同时减少总排班分钟数。
- 班次当前只记录时长，没有起止时间，因此还不能检查班次是否重叠。

`--sample` 使用固定的排班示例；`--input` 接收用户直接提供的结构化排班问题。对于 `--request` 生成的自然语言草稿，只有用户审阅后通过 `--solve-draft` 提交才会求解。

当前能源园区模型支持读取 8 个题目附件并完成问题一基准核算、问题二离散调度、问题三连续调度和问题五有来源的政策分析。问题四的离网运行及储能配置仍待实现；储能功率上限、SOC 初末条件和容量优化目标需要先明确。Evals 默认离线检查案例文件；只有显式使用 `--live` 才调用 DeepSeek 并消耗配额。
