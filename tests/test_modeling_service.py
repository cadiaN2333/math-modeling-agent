import pytest

from min_cost_flow_fixtures import EXPECTED_ARC_FLOWS, make_ready_analysis


def _service(*, analysis=None, registry=None):
    from math_modeling_agent.modeling_service import ModelingService

    selected_analysis = analysis or make_ready_analysis()
    return ModelingService(
        analyzer=lambda _request: selected_analysis,
        registry=registry,
    )


def test_service_requires_validated_user_confirmation_before_solving() -> None:
    from math_modeling_agent.modeling_service import ConfirmationToken

    class SpyRegistry:
        def __init__(self):
            self.select_calls = 0
            self.solve_calls = 0

        def select(self, _problem):
            self.select_calls += 1
            return object()

        def solve(self, _problem):
            self.solve_calls += 1
            raise AssertionError("未确认时不应调用求解器")

    registry = SpyRegistry()
    service = _service(registry=registry)
    session = service.create_draft("运输问题")

    with pytest.raises(PermissionError):
        service.solve_confirmed(
            ConfirmationToken(
                session_id=session.session_id,
                draft_hash=session.draft_hash,
                confirmed_at="2026-01-01T00:00:00+00:00",
                nonce="伪造-" + "x" * 40,
            )
        )

    assert registry.select_calls == 0
    assert registry.solve_calls == 0


def test_service_runs_draft_to_verified_state_and_returns_domain_result() -> None:
    service = _service()
    session = service.create_draft("将仓库货物低成本运到门店")

    validation = service.validate_draft(session.session_id, session.draft_hash)
    assert validation.valid
    assert service.get_session(session.session_id).state == "validated"

    token = service.confirm_draft(session.session_id, session.draft_hash)
    assert service.get_session(session.session_id).state == "confirmed"

    report = service.solve_confirmed(token)
    assert report.solver_status == "OPTIMAL"
    assert report.backend_id == "ortools_simple_min_cost_flow"
    assert report.result["arc_flows"] == EXPECTED_ARC_FLOWS
    assert service.get_session(session.session_id).state == "solved"

    verified = service.verify_result(session.session_id)
    assert verified.is_valid is True
    assert verified.validation_errors == []
    assert service.get_session(session.session_id).state == "verified"


def test_service_rejects_stale_draft_hash_before_confirmation() -> None:
    service = _service()
    session = service.create_draft("運輸問題")
    service.validate_draft(session.session_id, session.draft_hash)

    with pytest.raises(ValueError, match="草稿摘要"):
        service.confirm_draft(session.session_id, "過期摘要")

    assert service.get_session(session.session_id).state == "validated"


def test_service_returns_validation_error_when_domain_compilation_fails() -> None:
    analysis = make_ready_analysis()
    analysis.minimum_cost_flow_draft.arcs[0].from_node = "UNKNOWN"
    service = _service(analysis=analysis)
    session = service.create_draft("坏数据")

    result = service.validate_draft(session.session_id, session.draft_hash)

    assert not result.valid
    assert result.errors
    assert service.get_session(session.session_id).state == "draft"
    assert service.get_session(session.session_id).ir is None


def test_service_does_not_call_solver_when_registry_has_no_compatible_backend() -> None:
    class UnsupportedRegistry:
        def __init__(self):
            self.solve_calls = 0

        def select(self, _problem):
            return None

        def solve(self, _problem):
            self.solve_calls += 1
            raise AssertionError("不支持的模型不应交给求解器")

    registry = UnsupportedRegistry()
    service = _service(registry=registry)
    session = service.create_draft("运输问题")
    service.validate_draft(session.session_id, session.draft_hash)
    token = service.confirm_draft(session.session_id, session.draft_hash)

    report = service.solve_confirmed(token)

    assert report.solver_status == "UNSUPPORTED_MODEL"
    assert service.get_session(session.session_id).state == "confirmed"
    assert registry.solve_calls == 0


def test_returned_session_is_a_copy_and_cannot_mutate_internal_draft() -> None:
    service = _service()
    session = service.create_draft("运输问题")
    session.analysis.summary = "外部篡改"

    stored_session = service.get_session(session.session_id)

    assert stored_session.analysis.summary != "外部篡改"
    assert stored_session.draft_hash == session.draft_hash


def test_service_rejects_modified_confirmation_nonce_and_token_replay() -> None:
    from math_modeling_agent.modeling_service import ConfirmationToken

    service = _service()
    session = service.create_draft("运输问题")
    service.validate_draft(session.session_id, session.draft_hash)
    token = service.confirm_draft(session.session_id, session.draft_hash)
    tampered_token = token.model_copy(update={"nonce": "错误令牌"})

    with pytest.raises(PermissionError):
        service.solve_confirmed(tampered_token)

    report = service.solve_confirmed(token)
    assert report.solver_status == "OPTIMAL"

    with pytest.raises(PermissionError):
        service.solve_confirmed(token)

    assert isinstance(token, ConfirmationToken)
