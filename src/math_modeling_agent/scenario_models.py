"""定义基准线性模型和命名情景的输入契约。"""

from pydantic import BaseModel, Field, model_validator

from .models import LinearProgramProblem


class LinearScenario(BaseModel):
    """一个完整的线性模型情景。"""

    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    problem: LinearProgramProblem


class ScenarioAnalysisRequest(BaseModel):
    """包含基准模型和一个或多个情景模型的比较请求。"""

    base_problem: LinearProgramProblem
    scenarios: list[LinearScenario] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_scenarios(self) -> "ScenarioAnalysisRequest":
        """确保情景 ID 唯一且变量域/单位一致，差异只在模型参数中。"""

        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_id 必须唯一")

        base_signature = sorted(
            (variable.name, variable.unit, variable.domain)
            for variable in self.base_problem.variables
        )
        for scenario in self.scenarios:
            signature = sorted(
                (variable.name, variable.unit, variable.domain)
                for variable in scenario.problem.variables
            )
            if signature != base_signature:
                raise ValueError(
                    f"情景 {scenario.scenario_id} 的变量签名与基准模型不一致"
                )
        return self
