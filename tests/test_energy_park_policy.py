from test_energy_park_discrete import _dataset_with_sunny_half_day


def test_policy_report_has_evidence_for_impacts_and_recommendations() -> None:
    from math_modeling_agent.energy_park_continuous import (
        run_continuous_question_three,
    )
    from math_modeling_agent.energy_park_discrete import run_discrete_question_two
    from math_modeling_agent.energy_park_policy import build_energy_park_policy_report

    data = _dataset_with_sunny_half_day()
    discrete = run_discrete_question_two(data, target_levels_tons_per_day=[72])
    continuous = run_continuous_question_three(
        data,
        target_levels_tons_per_day=[72],
        discrete_result=discrete,
    )

    report = build_energy_park_policy_report(discrete, continuous)

    assert len(report.benefits) >= 3
    assert len(report.risks) >= 3
    assert len(report.recommendations) >= 3
    source_ids = {source.source_id for source in report.sources}
    assert all(
        claim.source_ids and set(claim.source_ids) <= source_ids
        for claim in [*report.benefits, *report.risks, *report.recommendations]
    )
    assert report.quantitative_comparisons[0].target_tons_per_day == 72
    assert any("配电网" in item for item in report.limitations)
