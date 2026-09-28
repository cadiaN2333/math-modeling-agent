# 排班数学建模 Agent

这是一个使用 OR-Tools 求解排班问题的可验证建模项目。当前版本包含结构化问题校验、CP-SAT 排班求解、独立结果验证，以及一个可运行的内置样例。

## 环境

- Python 3.11 或更高版本
- OR-Tools
- Pydantic
- pytest（开发和测试）

## 安装

在项目根目录 `D:\Agent` 的 PowerShell 终端中运行：

```powershell
python -m pip install -e ".[dev]"
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

## 运行测试

```powershell
python -m pytest -p no:cacheprovider -q
```

## HMML 方法检索

项目使用领域、子领域、方法节点三层结构保存建模知识，并在排班求解前根据问题描述和目标推荐方法。知识库范围、检索打分和当前支持边界见 docs/HMML设计说明.md。

## 使用 DeepSeek 分析自然语言问题

自然语言分析使用 DeepSeek V4.1 Flash；DeepSeek API 中对应的模型名是 deepseek-flash。模型输出会经过 Pydantic 结构校验，再为每个子任务检索 HMML。API 密钥只从环境变量读取，不要写进源码或提交到仓库。

在 PowerShell 中安装可选依赖并设置本窗口环境变量：

    python -m pip install -e ".[dev,agent]"
    $env:DEEPSEEK_API_KEY = "你的 DeepSeek API 密钥"
    $env:DEEPSEEK_MODEL = "deepseek-flash"
    python -m math_modeling_agent.cli --request "有三名员工，林晓有急救技能且最多工作八小时；安排两个八小时班次，每班至少一人，第一个班必须有急救技能。"

该命令会返回问题状态、已知条件、缺失信息、追问、子任务和 HMML 方法建议。当前阶段负责分析与拆分；把分析结果进一步转换成内部排班模型并自动求解，是后续步骤。

## 当前模型范围

- 每个班次有最低人员数要求。
- 班次可以要求一定数量的特定技能员工。
- 每名员工有最大总工时。
- 优化目标是满足约束的同时减少总排班分钟数。
- 班次当前只记录时长，没有起止时间，因此还不能检查班次是否重叠。

`--sample` 使用固定的示例数据；`--input` 允许提供自己的 JSON 排班问题。当前版本不连接外部模型 API，后续可以在此基础上加入自然语言需求解析。
