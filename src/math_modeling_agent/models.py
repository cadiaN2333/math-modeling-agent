from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Employee(BaseModel):
    employee_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    skills: set[str] = Field(default_factory=set)
    max_hours: float = Field(gt=0)

class Shift(BaseModel):
    shift_id: str = Field(min_length=1)
    duration_hours: float = Field(gt=0)


class CoverageRequirement(BaseModel):
    shift_id: str = Field(min_length=1)
    minimum_employees: int = Field(ge=0)
    required_skill_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("required_skill_counts")
    @classmethod
    def skill_counts_must_be_non_negative(
        cls,
        counts: dict[str, int],
    ) -> dict[str, int]:
        if any(count < 0 for count in counts.values()):
            raise ValueError("技能需求人数不能为负数")
        return counts


class SchedulingProblem(BaseModel):
    employees: list[Employee] = Field(min_length=1)
    shifts: list[Shift] = Field(min_length=1)
    coverage_requirements: list[CoverageRequirement]

    @model_validator(mode="after")
    def validate_references(self) -> "SchedulingProblem":
        employee_ids = [employee.employee_id for employee in self.employees]
        if len(employee_ids) != len(set(employee_ids)):
            raise ValueError("员工编号不能重复")

        shift_ids = [shift.shift_id for shift in self.shifts]
        if len(shift_ids) != len(set(shift_ids)):
            raise ValueError("班次编号不能重复")

        requirement_ids = [
            requirement.shift_id
            for requirement in self.coverage_requirements
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("每个班次只能有一条覆盖要求")

        known_shift_ids = set(shift_ids)
        if any(
            shift_id not in known_shift_ids
            for shift_id in requirement_ids
        ):
            raise ValueError("覆盖要求引用了不存在的班次")

        if set(requirement_ids) != known_shift_ids:
            raise ValueError("每个班次都必须有一条覆盖要求")

        return self


class FlowNode(BaseModel):
    """网络流中的节点；正供给为供货，负供给为需求，零表示中转。"""

    node_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    supply: int = Field(strict=True)


class FlowArc(BaseModel):
    """有向网络边，记录容量和每个流量单位的整数费用。"""

    arc_id: str = Field(min_length=1)
    from_node: str = Field(min_length=1)
    to_node: str = Field(min_length=1)
    capacity: int = Field(ge=0, strict=True)
    unit_cost: int = Field(strict=True)


class MinCostFlowProblem(BaseModel):
    """单商品、整数单位的最小费用网络流输入。"""

    nodes: list[FlowNode] = Field(min_length=2)
    arcs: list[FlowArc] = Field(min_length=1)
    flow_unit: str = Field(min_length=1)
    cost_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_references(self) -> "MinCostFlowProblem":
        """检查节点和边编号唯一、端点存在，并拒绝自连边。"""

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("节点编号不能重复")

        arc_ids = [arc.arc_id for arc in self.arcs]
        if len(arc_ids) != len(set(arc_ids)):
            raise ValueError("路线编号不能重复")

        known_node_ids = set(node_ids)
        for arc in self.arcs:
            if arc.from_node == arc.to_node:
                raise ValueError("路线不能连接节点自身")
            if arc.from_node not in known_node_ids or arc.to_node not in known_node_ids:
                raise ValueError("路线引用了不存在的节点")

        return self


class LinearVariable(BaseModel):
    """线性规划决策变量的名称、单位和变量域。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    domain: Literal["continuous", "integer", "binary"] = "continuous"


class LinearTerm(BaseModel):
    """线性表达式中的一个系数—变量项。"""

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(min_length=1)
    coefficient: float

    @field_validator("coefficient")
    @classmethod
    def coefficient_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("线性系数必须是有限数值")
        return value


class LinearObjective(BaseModel):
    """连续 LP 或 MILP 的单一线性目标函数。"""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["maximize", "minimize"]
    terms: list[LinearTerm] = Field(min_length=1)

    @model_validator(mode="after")
    def objective_variables_must_be_unique(self) -> "LinearObjective":
        variable_names = [term.variable for term in self.terms]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("同一目标表达式中变量不能重复")
        return self


class LinearConstraint(BaseModel):
    """连续 LP 或 MILP 的一条线性约束。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    terms: list[LinearTerm] = Field(min_length=1)
    relation: Literal["<=", ">=", "=="]
    rhs: float

    @field_validator("rhs")
    @classmethod
    def right_hand_side_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("约束右侧必须是有限数值")
        return value

    @model_validator(mode="after")
    def constraint_variables_must_be_unique(self) -> "LinearConstraint":
        variable_names = [term.variable for term in self.terms]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("同一约束表达式中变量不能重复")
        return self


class LinearProgramProblem(BaseModel):
    """连续、整数或二进制变量构成的单目标线性规划求解输入。"""

    model_config = ConfigDict(extra="forbid")

    variables: list[LinearVariable] = Field(min_length=1)
    objective: LinearObjective
    constraints: list[LinearConstraint]

    @model_validator(mode="after")
    def validate_variable_references(self) -> "LinearProgramProblem":
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("LP 变量名不能重复")

        constraint_names = [constraint.name for constraint in self.constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("LP 约束名称不能重复")

        known_variables = set(variable_names)
        referenced_variables = {
            term.variable
            for term in self.objective.terms
        }
        referenced_variables.update(
            term.variable
            for constraint in self.constraints
            for term in constraint.terms
        )
        unknown_variables = referenced_variables - known_variables
        if unknown_variables:
            raise ValueError(f"LP 表达式引用了未声明的变量：{sorted(unknown_variables)}")

        return self
