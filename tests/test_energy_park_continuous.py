import pytest

from test_energy_park_discrete import _dataset_with_sunny_half_day


def test_continuous_schedule_respects_minimum_fraction_and_target() -> None:
    from math_modeling_agent.energy_park_continuous import solve_continuous_schedule

    dataset = _dataset_with_sunny_half_day()
    run = solve_continuous_schedule(
        dataset,
        target_tons_per_day=36,
        scenario_id="sunny-half-day",
    )

    assert run.solver_status == "OPTIMAL"
    assert sum(run.process_fractions) == pytest.approx(12)
    assert all(0.1 <= value <= 1 for value in run.process_fractions)
    assert run.operation.ammonia_production_tons == pytest.approx(36)
    assert run.validation_report.is_valid is True
    assert all(
        buy <= 1e-7 or sell <= 1e-7
        for buy, sell in zip(
            run.solver_grid_purchase_mw,
            run.solver_grid_export_mw,
        )
    )


def test_continuous_schedule_rejects_target_below_minimum_load() -> None:
    from math_modeling_agent.energy_park_continuous import solve_continuous_schedule

    with pytest.raises(ValueError, match="10% 最低负荷"):
        solve_continuous_schedule(
            _dataset_with_sunny_half_day(),
            target_tons_per_day=6,
            scenario_id="below-minimum",
        )


def test_question_three_compares_72_ton_continuous_schedule_with_discrete() -> None:
    from math_modeling_agent.energy_park_continuous import (
        run_continuous_question_three,
    )

    result = run_continuous_question_three(
        _dataset_with_sunny_half_day(),
        target_levels_tons_per_day=[72],
    )

    assert len(result.typical_runs) == 1
    assert len(result.scenario_runs) == 24
    assert result.typical_runs[0].process_fractions == pytest.approx([1.0] * 24)
    comparison = result.discrete_comparison_by_target[0]
    assert comparison["target_tons_per_day"] == 72
    assert comparison["annual_cost_per_ton_difference_yuan"] == pytest.approx(0, abs=1e-5)
