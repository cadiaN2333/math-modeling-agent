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
