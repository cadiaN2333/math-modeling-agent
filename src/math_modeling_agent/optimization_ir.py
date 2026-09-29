from math import isfinite
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from math_modeling_agent.models import MinCostFlowProblem


class _IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("source_fact_ids", check_fields=False)
    @classmethod
    def source_fact_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("source_fact_ids 不能包含重复引用")
        return value


class EvidenceIR(_IRModel):
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    title: str = Field(min_length=1)


class ModelAssumption(_IRModel):
    assumption_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    source_fact_ids: list[str] = Field(default_factory=list)


class LinearVariableIR(_IRModel):
    variable_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    domain: Literal["continuous", "integer", "binary"]
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str = Field(min_length=1)
    source_fact_ids: list[str] = Field(default_factory=list)

    @field_validator("lower_bound", "upper_bound")
    @classmethod
    def bounds_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("变量上下界必须是有限数值")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "LinearVariableIR":
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("变量上下界不能反转")
        if self.domain == "binary":
            if self.lower_bound is not None and self.lower_bound < 0:
                raise ValueError("二进制变量下界不能小于 0")
            if self.upper_bound is not None and self.upper_bound > 1:
                raise ValueError("二进制变量上界不能大于 1")
        return self


class LinearTermIR(_IRModel):
    variable_id: str = Field(min_length=1)
    coefficient: float

    @field_validator("coefficient")
    @classmethod
    def coefficient_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("线性系数必须是有限数值")
        return value


class LinearObjectiveIR(_IRModel):
    direction: Literal["maximize", "minimize"]
    terms: list[LinearTermIR] = Field(min_length=1)
    constant: float = 0
    unit: str | None = Field(default=None, min_length=1)

    @field_validator("constant")
    @classmethod
    def constant_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("目标常数项必须是有限数值")
        return value

    @model_validator(mode="after")
    def objective_variables_must_be_unique(self) -> "LinearObjectiveIR":
        variable_ids = [term.variable_id for term in self.terms]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("目标表达式中的变量 ID 不能重复")
        return self


class LinearConstraintIR(_IRModel):
    constraint_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    terms: list[LinearTermIR] = Field(min_length=1)
    relation: Literal["<=", ">=", "=="]
    rhs: float
    constant: float = 0
    unit: str | None = Field(default=None, min_length=1)
    source_fact_ids: list[str] = Field(default_factory=list)

    @field_validator("rhs", "constant")
    @classmethod
    def constraint_values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("约束数值必须是有限数值")
        return value

    @model_validator(mode="after")
    def constraint_variables_must_be_unique(self) -> "LinearConstraintIR":
        variable_ids = [term.variable_id for term in self.terms]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("约束表达式中的变量 ID 不能重复")
        return self


class LinearFormulationIR(_IRModel):
    kind: Literal["linear"]
    variables: list[LinearVariableIR] = Field(min_length=1)
    objective: LinearObjectiveIR
    constraints: list[LinearConstraintIR] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids_and_references(self) -> "LinearFormulationIR":
        variable_ids = [variable.variable_id for variable in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("决策变量 ID 不能重复")

        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("约束 ID 不能重复")

        declared_variable_ids = set(variable_ids)
        referenced_variable_ids = {
            term.variable_id for term in self.objective.terms
        }
        referenced_variable_ids.update(
            term.variable_id
            for constraint in self.constraints
            for term in constraint.terms
        )
        undeclared_variable_ids = referenced_variable_ids - declared_variable_ids
        if undeclared_variable_ids:
            raise ValueError(
                f"表达式引用了未声明变量 ID：{sorted(undeclared_variable_ids)}"
            )
        return self


class NetworkFlowNodeIR(_IRModel):
    node_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    supply: int = Field(strict=True)
    source_fact_ids: list[str] = Field(default_factory=list)


class NetworkFlowArcIR(_IRModel):
    arc_id: str = Field(min_length=1)
    from_node: str = Field(min_length=1)
    to_node: str = Field(min_length=1)
    capacity: int = Field(ge=0, strict=True)
    unit_cost: int = Field(strict=True)
    source_fact_ids: list[str] = Field(default_factory=list)


class NetworkFlowFormulationIR(_IRModel):
    kind: Literal["network_flow"]
    nodes: list[NetworkFlowNodeIR] = Field(min_length=2)
    arcs: list[NetworkFlowArcIR] = Field(min_length=1)
    flow_unit: str = Field(min_length=1)
    cost_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_network_with_domain_rules(self) -> "NetworkFlowFormulationIR":
        MinCostFlowProblem.model_validate(
            {
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "name": node.name,
                        "supply": node.supply,
                    }
                    for node in self.nodes
                ],
                "arcs": [
                    {
                        "arc_id": arc.arc_id,
                        "from_node": arc.from_node,
                        "to_node": arc.to_node,
                        "capacity": arc.capacity,
                        "unit_cost": arc.unit_cost,
                    }
                    for arc in self.arcs
                ],
                "flow_unit": self.flow_unit,
                "cost_unit": self.cost_unit,
            }
        )
        return self


FormulationIR = Annotated[
    Union[LinearFormulationIR, NetworkFlowFormulationIR],
    Field(discriminator="kind"),
]


class OptimizationIR(_IRModel):
    schema_version: Literal["1"]
    problem_id: str = Field(min_length=1)
    problem_family: Literal[
        "employee_scheduling",
        "linear_programming",
        "minimum_cost_flow",
        "energy_park",
    ]
    formulation: FormulationIR
    evidence: list[EvidenceIR]
    assumptions: list[ModelAssumption]

    @model_validator(mode="after")
    def validate_families_evidence_and_assumptions(self) -> "OptimizationIR":
        is_network_flow = isinstance(self.formulation, NetworkFlowFormulationIR)
        if (self.problem_family == "minimum_cost_flow") != is_network_flow:
            raise ValueError("问题类型与 formulation.kind 不匹配")

        evidence_ids = [item.source_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("证据 source_id 不能重复")

        assumption_ids = [item.assumption_id for item in self.assumptions]
        if len(assumption_ids) != len(set(assumption_ids)):
            raise ValueError("假设 ID 不能重复")

        known_evidence_ids = set(evidence_ids)
        source_lists: list[list[str]] = [
            assumption.source_fact_ids for assumption in self.assumptions
        ]
        if isinstance(self.formulation, LinearFormulationIR):
            source_lists.extend(
                variable.source_fact_ids for variable in self.formulation.variables
            )
            source_lists.extend(
                constraint.source_fact_ids
                for constraint in self.formulation.constraints
            )
        else:
            source_lists.extend(node.source_fact_ids for node in self.formulation.nodes)
            source_lists.extend(arc.source_fact_ids for arc in self.formulation.arcs)

        for source_ids in source_lists:
            unknown_source_ids = set(source_ids) - known_evidence_ids
            if unknown_source_ids:
                raise ValueError(
                    f"引用了不存在的证据 source_id：{sorted(unknown_source_ids)}"
                )
        return self
