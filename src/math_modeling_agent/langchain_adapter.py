"""将可选 LangChain Agent 接到与框架无关的 ModelingService。"""

import os
from dataclasses import asdict
from typing import Any, Callable

from .analysis_agent import ANALYSIS_INSTRUCTIONS, ProblemAnalysis, load_project_environment
from .knowledge_models import RetrievedEvidence
from .modeling_service import (
    ConfirmationToken,
    DraftValidation,
    ModelingReport,
    ModelingService,
    ModelingSession,
)


def create_deepseek_chat_model(
    *,
    chat_model_factory: Callable[..., Any] | None = None,
) -> Any:
    """从项目环境配置构造 LangChain DeepSeek V4.1 Flash 模型。"""

    load_project_environment()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY 环境变量")

    if chat_model_factory is None:
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as error:
            raise RuntimeError(
                '缺少 LangChain DeepSeek 依赖；请运行 python -m pip install -e ".[langchain]"'
            ) from error
        chat_model_factory = ChatDeepSeek

    return chat_model_factory(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        max_retries=2,
        # 关闭思考模式，避免工具调用往返依赖 reasoning_content 回传。
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
    )


class LangChainModelingAdapter:
    """把 LangChain Agent 的结构化分析结果交给公共建模服务。"""

    def __init__(
        self,
        agent: Any,
        service: ModelingService,
        retrieved_evidence: list[RetrievedEvidence] | None = None,
    ) -> None:
        self._agent = agent
        self._service = service
        self._retrieved_evidence = retrieved_evidence if retrieved_evidence is not None else []
        self._evidence_by_session: dict[str, list[RetrievedEvidence]] = {}

    def create_draft(self, request: str) -> ModelingSession:
        """运行 Agent 生成草稿；仅保存结果，不确认、不求解。"""

        self._retrieved_evidence.clear()
        try:
            state = self._agent.invoke(
                {"messages": [{"role": "user", "content": request}]}
            )
        except Exception as error:
            raise RuntimeError(
                f"LangChain Agent 调用失败（{type(error).__name__}）"
            ) from error
        structured = state.get("structured_response")
        if structured is None:
            raise RuntimeError("LangChain Agent 没有返回结构化建模草稿")
        analysis = (
            structured
            if isinstance(structured, ProblemAnalysis)
            else ProblemAnalysis.model_validate(structured)
        )
        session = self._service.create_draft_from_analysis(analysis)
        unique_evidence = {
            item.chunk_id: item.model_copy(deep=True)
            for item in self._retrieved_evidence
        }
        self._evidence_by_session[session.session_id] = list(unique_evidence.values())
        return session

    def get_evidence(self, session_id: str) -> list[RetrievedEvidence]:
        """返回 Agent 在该草稿分析中使用过的来源证据。"""

        return [
            item.model_copy(deep=True)
            for item in self._evidence_by_session.get(session_id, [])
        ]

    def validate_draft(self, session_id: str, draft_hash: str) -> DraftValidation:
        return self._service.validate_draft(session_id, draft_hash)

    def confirm_draft(self, session_id: str, draft_hash: str) -> ConfirmationToken:
        """由 UI 在用户明确确认后调用；此方法不是 Agent 工具。"""

        return self._service.confirm_draft(session_id, draft_hash)

    def solve_confirmed(self, token: ConfirmationToken) -> ModelingReport:
        """由 UI/应用服务在确认后调用；此方法不是 Agent 工具。"""

        return self._service.solve_confirmed(token)

    def verify_result(self, session_id: str) -> ModelingReport:
        return self._service.verify_result(session_id)


def create_langchain_modeling_agent(
    service: ModelingService,
    *,
    model: Any | None = None,
    create_agent_factory: Callable[..., Any] | None = None,
    tool_decorator: Callable[[Callable[..., Any]], Any] | None = None,
    tool_strategy_factory: Callable[[type], Any] | None = None,
) -> LangChainModelingAdapter:
    """构造结构化问题分析 Agent，并只暴露只读方法/RAG 检索工具。"""

    if (
        create_agent_factory is None
        or tool_decorator is None
        or tool_strategy_factory is None
    ):
        try:
            from langchain.agents import create_agent
            from langchain.agents.structured_output import ToolStrategy
            from langchain.tools import tool
        except ImportError as error:
            raise RuntimeError(
                '缺少 LangChain 依赖；请运行 python -m pip install -e ".[langchain]"'
            ) from error
        create_agent_factory = create_agent
        tool_decorator = tool
        tool_strategy_factory = ToolStrategy

    active_model = model if model is not None else create_deepseek_chat_model()
    tools: list[Any] = []
    retrieved_evidence: list[RetrievedEvidence] = []

    @tool_decorator
    def search_modeling_methods(
        problem_description: str,
        desired_outcome: str,
        required_method_id: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        """按问题和目标从 HMML 方法树检索可用数学建模方法。"""

        recommendations = service.search_methods(
            problem_description,
            desired_outcome,
            required_method_id=required_method_id,
            top_k=top_k,
        )
        return [asdict(item) for item in recommendations]

    tools.append(search_modeling_methods)

    if service.has_knowledge_service:

        @tool_decorator
        def search_modeling_knowledge(
            query: str,
            problem_family: str,
            solver_ids: list[str] | None = None,
            top_k: int = 5,
        ) -> list[dict[str, object]]:
            """检索审核通过的带来源知识；返回数字只可作为待确认参考。"""

            evidence: list[RetrievedEvidence] = service.search_knowledge(
                query,
                problem_family=problem_family,
                solver_ids=solver_ids,
                top_k=top_k,
            )
            retrieved_evidence.extend(evidence)
            return [item.model_dump(mode="json") for item in evidence]

        tools.append(search_modeling_knowledge)

    agent = create_agent_factory(
        model=active_model,
        tools=tools,
        system_prompt=(
            f"{ANALYSIS_INSTRUCTIONS}\n\n"
            "检索到的知识仅用于方法解释和带来源的参考；不能把检索数字或默认假设写成用户给出的事实。"
            "不要调用或模拟用户确认；最终只输出结构化 ProblemAnalysis 草稿。"
        ),
        response_format=tool_strategy_factory(ProblemAnalysis),
    )
    return LangChainModelingAdapter(agent, service, retrieved_evidence)
