def test_hmml_contains_domain_subdomain_and_method_levels() -> None:
    from math_modeling_agent.hmml import load_hmml

    library = load_hmml()

    assert library.domains
    assert any(domain.subdomains for domain in library.domains)
    assert any(
        subdomain.methods
        for domain in library.domains
        for subdomain in domain.subdomains
    )


def test_retrieval_ranks_cp_sat_for_employee_scheduling() -> None:
    from math_modeling_agent.retriever import HMMLRetriever

    retriever = HMMLRetriever()
    recommendations = retriever.retrieve(
        problem_description="给员工安排班次，每班要满足急救技能和最低人数，并遵守工时上限。",
        desired_outcome="找到满足约束且总排班时间较少的可行方案。",
        top_k=3,
    )

    assert recommendations
    assert recommendations[0].method_id == "cp_sat_scheduling"
    assert recommendations[0].domain_name == "运筹优化"
    assert recommendations[0].subdomain_name == "排班与资源分配"
    assert recommendations[0].implementation_status == "已实现"
    assert recommendations[0].problem_score > 0
    assert recommendations[0].goal_score > 0


def test_retrieval_ranks_scip_for_mixed_integer_linear_programming() -> None:
    from math_modeling_agent.linear_agent import _describe_linear_program
    from math_modeling_agent.models import (
        LinearObjective,
        LinearProgramProblem,
        LinearTerm,
        LinearVariable,
    )
    from math_modeling_agent.retriever import HMMLRetriever

    problem = LinearProgramProblem(
        variables=[LinearVariable(name="x", unit="件", domain="integer")],
        objective=LinearObjective(
            direction="maximize",
            terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[],
    )
    recommendations = HMMLRetriever().retrieve(
        problem_description=_describe_linear_program(problem),
        desired_outcome="使用整数/混合整数线性规划和 SCIP 满足线性约束并优化单一目标函数。",
        top_k=3,
    )

    integer_method = next(
        item for item in recommendations if item.method_id == "integer_programming"
    )
    assert integer_method.implementation_status == "已实现"
    assert integer_method.solver == "OR-Tools SCIP"


def test_retrieval_can_require_compatible_method_beyond_top_k() -> None:
    from math_modeling_agent.retriever import HMMLRetriever

    recommendations = HMMLRetriever().retrieve(
        problem_description="生产计划与资源约束",
        desired_outcome="最大化产量",
        top_k=1,
        required_method_id="integer_programming",
    )

    assert len(recommendations) == 1
    assert recommendations[0].method_id == "integer_programming"
    assert recommendations[0].solver == "OR-Tools SCIP"


def test_retrieval_returns_no_candidates_for_unrelated_text() -> None:
    from math_modeling_agent.retriever import HMMLRetriever

    recommendations = HMMLRetriever().retrieve(
        problem_description="介绍一下唐朝的诗歌。",
        desired_outcome="写一首五言绝句。",
    )

    assert recommendations == []


def test_retrieval_requires_positive_top_k() -> None:
    import pytest

    from math_modeling_agent.retriever import HMMLRetriever

    with pytest.raises(ValueError, match="top_k"):
        HMMLRetriever().retrieve("员工排班", top_k=0)
