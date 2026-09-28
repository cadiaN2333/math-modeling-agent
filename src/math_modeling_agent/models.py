from pydantic import BaseModel, Field, field_validator, model_validator


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