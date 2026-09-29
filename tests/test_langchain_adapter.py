import pytest

from min_cost_flow_fixtures import make_ready_analysis


def test_langchain_adapter_module_imports_without_optional_langchain_packages() -> None:
    import math_modeling_agent.langchain_adapter as adapter

    assert callable(adapter.create_langchain_modeling_agent)


def test_agent_uses_structured_analysis_rag_tools_and_cannot_confirm_or_solve() -> None:
    from math_modeling_agent.langchain_adapter import (
        create_langchain_modeling_agent,
    )
    from math_modeling_agent.modeling_service import ModelingService

    analysis = make_ready_analysis()

    class FakeKnowledgeService:
        def retrieve(self, query, **filters):
            return []

        def retrieve_methods(self, problem_description, desired_outcome, **filters):
            return []

    class FakeAgentGraph:
        def __init__(self):
            self.inputs = []

        def invoke(self, value):
            self.inputs.append(value)
            return {"structured_response": analysis}

    created = {}
    graph = FakeAgentGraph()

    def fake_tool(function):
        function.name = function.__name__
        return function

    class FakeToolStrategy:
        def __init__(self, schema):
            self.schema = schema

    def fake_create_agent(**kwargs):
        created.update(kwargs)
        return graph

    service = ModelingService(
        analyzer=lambda _request: analysis,
        knowledge_service=FakeKnowledgeService(),
    )
    adapter = create_langchain_modeling_agent(
        service,
        model=object(),
        create_agent_factory=fake_create_agent,
        tool_decorator=fake_tool,
        tool_strategy_factory=FakeToolStrategy,
    )

    session = adapter.create_draft("以最低费用满足仓库供需")

    assert session.state == "draft"
    assert session.analysis == analysis
    assert graph.inputs == [
        {"messages": [{"role": "user", "content": "以最低费用满足仓库供需"}]}
    ]
    assert created["response_format"].schema.__name__ == "ProblemAnalysis"
    tool_names = {tool.name for tool in created["tools"]}
    assert tool_names == {"search_modeling_methods", "search_modeling_knowledge"}
    assert "confirm_draft" not in tool_names
    assert "solve_confirmed" not in tool_names


def test_agent_uses_methods_tool_without_configured_rag() -> None:
    from math_modeling_agent.langchain_adapter import create_langchain_modeling_agent
    from math_modeling_agent.modeling_service import ModelingService

    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    adapter = create_langchain_modeling_agent(
        ModelingService(analyzer=lambda _request: make_ready_analysis()),
        model=object(),
        create_agent_factory=fake_create_agent,
        tool_decorator=lambda function: setattr(function, "name", function.__name__) or function,
        tool_strategy_factory=lambda schema: schema,
    )

    assert adapter is not None
    assert {tool.name for tool in captured["tools"]} == {"search_modeling_methods"}


def test_deepseek_model_factory_uses_project_settings_without_exposing_secret(monkeypatch) -> None:
    from math_modeling_agent.langchain_adapter import create_deepseek_chat_model

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret-not-real")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __repr__(self):
            return "FakeChatDeepSeek()"

    model = create_deepseek_chat_model(chat_model_factory=FakeChatDeepSeek)

    assert isinstance(model, FakeChatDeepSeek)
    assert captured["model"] == "deepseek-flash"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["model_kwargs"] == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert "unit-test-secret-not-real" not in repr(model)


def test_agent_invocation_error_does_not_echo_underlying_exception_text() -> None:
    from math_modeling_agent.langchain_adapter import LangChainModelingAdapter
    from math_modeling_agent.modeling_service import ModelingService

    class FailingAgent:
        def invoke(self, _request):
            raise RuntimeError("request failed with sk-secret-value")

    adapter = LangChainModelingAdapter(
        FailingAgent(),
        ModelingService(analyzer=lambda _request: make_ready_analysis()),
    )

    with pytest.raises(RuntimeError, match="RuntimeError") as error:
        adapter.create_draft("运输问题")

    assert "sk-secret-value" not in str(error.value)
