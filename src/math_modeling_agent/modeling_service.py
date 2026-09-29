"""管理优化草稿确认、求解与领域独立验证。"""

import hashlib
import hmac
import json
import secrets
import uuid
from datetime import UTC, datetime
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .analysis_agent import (
    ProblemAnalysis,
    analyze_problem,
    to_linear_program_problem,
    to_minimum_cost_flow_problem,
    to_scheduling_problem,
)
from .linear_validator import validate_linear_solution
from .knowledge_models import RetrievedEvidence
from .knowledge_service import KnowledgeService
from .min_cost_flow_validator import validate_min_cost_flow_solution
from .optimization_compilers import (
    compile_linear_problem,
    compile_min_cost_flow_problem,
    compile_scheduling_problem,
    decode_flow_solution,
    decode_schedule_solution,
)
from .optimization_ir import EvidenceIR, OptimizationIR
from .optimization_registry import OptimizationResult, SolverBackend, SolverRegistry
from .retriever import HMMLRetriever, MethodRecommendation
from .solver import Assignment
from .validator import validate_solution


SessionState = Literal["draft", "validated", "confirmed", "solved", "verified"]
ProblemFamily = Literal[
    "employee_scheduling",
    "linear_programming",
    "minimum_cost_flow",
]


class DraftValidation(BaseModel):
    """草稿结构检查结果以及检查时的内容摘要。"""

    valid: bool
    errors: list[str]
    draft_hash: str


class ConfirmationToken(BaseModel):
    """由服务端签发的单次确认凭据；不得暴露为 Agent 工具参数。"""

    session_id: str
    draft_hash: str
    confirmed_at: str
    nonce: str = Field(min_length=32)


class ModelingSession(BaseModel):
    """服务端保存的建模会话快照。"""

    session_id: str
    state: SessionState
    draft_hash: str
    analysis: ProblemAnalysis
    ir: OptimizationIR | None = None


class ModelingReport(BaseModel):
    """统一报告，同时保留原问题族易读的解结构。"""

    session_id: str
    problem_id: str
    problem_family: ProblemFamily
    solver_status: str
    backend_id: str
    objective_value: float | int | None = None
    variable_values: dict[str, float | int] = Field(default_factory=dict)
    result: dict[str, object] = Field(default_factory=dict)
    evidence: list[EvidenceIR] = Field(default_factory=list)
    is_valid: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)


class ModelingService:
    """不依赖 LangChain 的状态机服务；当前会话存储仅驻留内存。"""

    def __init__(
        self,
        *,
        analyzer: Callable[[str], ProblemAnalysis] | None = None,
        registry: SolverRegistry | None = None,
        knowledge_service: KnowledgeService | None = None,
    ) -> None:
        self._analyzer = analyzer or analyze_problem
        self._registry = registry or SolverRegistry()
        self._knowledge_service = knowledge_service
        self._sessions: dict[str, ModelingSession] = {}
        self._confirmation_digests: dict[str, str] = {}
        self._results: dict[str, OptimizationResult] = {}
        self._reports: dict[str, ModelingReport] = {}

    @staticmethod
    def _analysis_hash(analysis: ProblemAnalysis) -> str:
        """按稳定 JSON 表示计算草稿摘要。"""

        canonical_json = json.dumps(
            analysis.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def _get_internal_session(self, session_id: str) -> ModelingSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"不存在建模会话：{session_id}") from error

    @staticmethod
    def _copy_session(session: ModelingSession) -> ModelingSession:
        return session.model_copy(deep=True)

    def create_draft(self, request: str) -> ModelingSession:
        """分析自然语言请求并创建待审阅草稿；此阶段不编译或求解。"""

        if not request.strip():
            raise ValueError("问题描述不能为空")
        return self.create_draft_from_analysis(self._analyzer(request))

    def create_draft_from_analysis(
        self,
        analysis: ProblemAnalysis,
    ) -> ModelingSession:
        """从已审阅文件或离线 Evals 结果创建草稿，不重新调用语言模型。"""

        analysis = analysis.model_copy(deep=True)
        session_id = str(uuid.uuid4())
        session = ModelingSession(
            session_id=session_id,
            state="draft",
            draft_hash=self._analysis_hash(analysis),
            analysis=analysis,
        )
        self._sessions[session_id] = session
        return self._copy_session(session)

    def get_session(self, session_id: str) -> ModelingSession:
        """返回会话副本，避免调用方通过可变模型修改服务端草稿。"""

        return self._copy_session(self._get_internal_session(session_id))

    def retrieve_knowledge(
        self,
        session_id: str,
        *,
        query: str | None = None,
        solver_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """按问题族查询带来源的参考知识，不改写或补充题目事实。"""

        analysis = self._get_internal_session(session_id).analysis
        query_text = query or " ".join(
            [
                analysis.summary,
                *analysis.known_facts,
                *(item.hmml_problem_query for item in analysis.subtasks),
                *(item.hmml_goal_query for item in analysis.subtasks),
            ]
        )
        if solver_ids is None:
            if analysis.problem_family == "employee_scheduling":
                solver_ids = ["ortools_cp_sat", "ortools_scip"]
            elif analysis.problem_family == "minimum_cost_flow":
                solver_ids = ["ortools_simple_min_cost_flow"]
            elif analysis.problem_family == "linear_programming":
                has_discrete_variables = any(
                    variable.domain != "continuous"
                    for variable in analysis.linear_program_draft.variables
                )
                solver_ids = (
                    ["ortools_scip"]
                    if has_discrete_variables
                    else ["ortools_glop"]
                )

        return self.search_knowledge(
            query_text,
            problem_family=analysis.problem_family,
            solver_ids=solver_ids,
            top_k=top_k,
        )

    @property
    def has_knowledge_service(self) -> bool:
        """指示是否已显式配置带来源的 RAG 检索。"""

        return self._knowledge_service is not None

    def search_knowledge(
        self,
        query: str,
        *,
        problem_family: str,
        solver_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        """供只读工具和服务会话共享的知识证据检索入口。"""

        if self._knowledge_service is None:
            raise RuntimeError("当前 ModelingService 未配置知识检索服务")
        return self._knowledge_service.retrieve(
            query,
            problem_family=problem_family,
            solver_ids=solver_ids,
            top_k=top_k,
        )

    def search_methods(
        self,
        problem_description: str,
        desired_outcome: str,
        *,
        required_method_id: str | None = None,
        top_k: int = 3,
    ) -> list[MethodRecommendation]:
        """通过 HMML 结构化关键词知识检索方法建议。"""

        if self._knowledge_service is not None:
            return self._knowledge_service.retrieve_methods(
                problem_description,
                desired_outcome,
                required_method_id=required_method_id,
                top_k=top_k,
            )
        return HMMLRetriever().retrieve(
            problem_description=problem_description,
            desired_outcome=desired_outcome,
            required_method_id=required_method_id,
            top_k=top_k,
        )

    def _compile_analysis(self, session: ModelingSession) -> OptimizationIR:
        analysis = session.analysis
        if analysis.status != "ready":
            raise ValueError("问题分析尚未就绪，不能编译优化模型")

        if analysis.problem_family == "employee_scheduling":
            problem = to_scheduling_problem(analysis)
            return compile_scheduling_problem(
                problem,
                problem_id=session.session_id,
            )
        if analysis.problem_family == "linear_programming":
            problem = to_linear_program_problem(analysis)
            return compile_linear_problem(
                problem,
                problem_id=session.session_id,
            )
        if analysis.problem_family == "minimum_cost_flow":
            problem = to_minimum_cost_flow_problem(analysis)
            return compile_min_cost_flow_problem(
                problem,
                problem_id=session.session_id,
            )
        raise ValueError(f"当前问题类型不支持：{analysis.problem_family}")

    def validate_draft(
        self,
        session_id: str,
        draft_hash: str,
    ) -> DraftValidation:
        """检查分析状态并编译 IR，但不调用求解器。"""

        session = self._get_internal_session(session_id)
        if session.state not in {"draft", "validated"}:
            raise ValueError("当前会话状态不允许重新校验草稿")

        current_hash = self._analysis_hash(session.analysis)
        if current_hash != session.draft_hash:
            session.state = "draft"
            session.ir = None
            session.draft_hash = current_hash
            return DraftValidation(
                valid=False,
                errors=["草稿内容已变化，请重新审阅并校验。"],
                draft_hash=current_hash,
            )
        if draft_hash != session.draft_hash:
            return DraftValidation(
                valid=False,
                errors=["草稿摘要不匹配，请刷新草稿后重试。"],
                draft_hash=session.draft_hash,
            )
        if session.analysis.status != "ready":
            if session.analysis.status == "needs_clarification":
                errors = list(session.analysis.clarifying_questions)
            else:
                errors = list(session.analysis.unsupported_reasons)
            if not errors:
                errors = ["分析结果尚未达到可建模状态。"]
            session.ir = None
            return DraftValidation(
                valid=False,
                errors=errors,
                draft_hash=session.draft_hash,
            )

        try:
            session.ir = self._compile_analysis(session)
        except (ValueError, TypeError) as error:
            session.ir = None
            session.state = "draft"
            return DraftValidation(
                valid=False,
                errors=[str(error)],
                draft_hash=session.draft_hash,
            )
        except Exception as error:
            # Pydantic 验证异常和领域构造错误都留在草稿阶段，不触发求解。
            session.ir = None
            session.state = "draft"
            return DraftValidation(
                valid=False,
                errors=[f"模型编译失败：{error}"],
                draft_hash=session.draft_hash,
            )

        session.state = "validated"
        return DraftValidation(
            valid=True,
            errors=[],
            draft_hash=session.draft_hash,
        )

    def confirm_draft(
        self,
        session_id: str,
        draft_hash: str,
    ) -> ConfirmationToken:
        """在调用方收到真实用户确认后签发单次求解凭据。"""

        session = self._get_internal_session(session_id)
        if session.state != "validated" or session.ir is None:
            raise ValueError("只有通过校验的草稿才能确认")
        if (
            draft_hash != session.draft_hash
            or self._analysis_hash(session.analysis) != session.draft_hash
        ):
            raise ValueError("草稿摘要已变化，必须重新校验后再确认")

        nonce = secrets.token_urlsafe(32)
        token = ConfirmationToken(
            session_id=session_id,
            draft_hash=session.draft_hash,
            confirmed_at=datetime.now(UTC).isoformat(),
            nonce=nonce,
        )
        self._confirmation_digests[session_id] = hashlib.sha256(
            nonce.encode("utf-8")
        ).hexdigest()
        session.state = "confirmed"
        return token

    def _require_valid_confirmation(
        self,
        token: ConfirmationToken,
    ) -> ModelingSession:
        session = self._get_internal_session(token.session_id)
        stored_digest = self._confirmation_digests.get(token.session_id)
        supplied_digest = hashlib.sha256(token.nonce.encode("utf-8")).hexdigest()
        if (
            session.state != "confirmed"
            or stored_digest is None
            or not hmac.compare_digest(stored_digest, supplied_digest)
            or token.draft_hash != session.draft_hash
            or self._analysis_hash(session.analysis) != session.draft_hash
        ):
            raise PermissionError("确认凭据无效、已过期或已被使用")
        return session

    def solve_confirmed(self, token: ConfirmationToken) -> ModelingReport:
        """仅凭服务端签发且未使用的确认凭据调用兼容后端。"""

        session = self._require_valid_confirmation(token)
        if session.ir is None:
            raise ValueError("会话中没有已校验的 IR 模型")

        backend: SolverBackend | None = self._registry.select(session.ir)
        if backend is None:
            report = ModelingReport(
                session_id=session.session_id,
                problem_id=session.ir.problem_id,
                problem_family=session.analysis.problem_family,
                solver_status="UNSUPPORTED_MODEL",
                backend_id="none",
                evidence=session.ir.evidence,
                validation_errors=["没有兼容的优化求解后端。"],
            )
            self._reports[session.session_id] = report
            return report.model_copy(deep=True)

        solver_result = backend.solve(session.ir)
        self._results[session.session_id] = solver_result
        report = self._build_report(session, solver_result)
        self._reports[session.session_id] = report

        if solver_result.status not in {"UNSUPPORTED_MODEL", "SOLVER_UNAVAILABLE", "MODEL_INVALID"}:
            session.state = "solved"
        return report.model_copy(deep=True)

    @staticmethod
    def _build_report(
        session: ModelingSession,
        solver_result: OptimizationResult,
    ) -> ModelingReport:
        """把 IR 解映射成用户易读的领域结果。"""

        analysis = session.analysis
        problem_id = session.ir.problem_id if session.ir is not None else session.session_id
        if solver_result.status not in {"OPTIMAL", "FEASIBLE"}:
            return ModelingReport(
                session_id=session.session_id,
                problem_id=problem_id,
                problem_family=analysis.problem_family,
                solver_status=solver_result.status,
                backend_id=solver_result.backend_id,
                objective_value=None,
                variable_values={},
                result={},
                evidence=session.ir.evidence if session.ir is not None else [],
            )

        result: dict[str, object]
        if analysis.problem_family == "employee_scheduling":
            problem = to_scheduling_problem(analysis)
            assignments = decode_schedule_solution(
                problem,
                solver_result.variable_values,
            )
            objective_minutes = (
                int(round(solver_result.objective_value))
                if solver_result.objective_value is not None
                else None
            )
            result = {
                "assignments": [
                    {
                        "employee_id": assignment.employee_id,
                        "shift_id": assignment.shift_id,
                    }
                    for assignment in assignments
                ],
                "objective_minutes": objective_minutes,
            }
        elif analysis.problem_family == "linear_programming":
            result = {
                "variable_values": dict(solver_result.variable_values),
                "objective_value": solver_result.objective_value,
            }
        else:
            problem = to_minimum_cost_flow_problem(analysis)
            arc_flows = decode_flow_solution(
                problem,
                solver_result.variable_values,
            )
            result = {
                "arc_flows": arc_flows,
                "total_cost": solver_result.objective_value,
            }

        return ModelingReport(
            session_id=session.session_id,
            problem_id=problem_id,
            problem_family=analysis.problem_family,
            solver_status=solver_result.status,
            backend_id=solver_result.backend_id,
            objective_value=solver_result.objective_value,
            variable_values=dict(solver_result.variable_values),
            result=result,
            evidence=session.ir.evidence if session.ir is not None else [],
        )

    def verify_result(self, session_id: str) -> ModelingReport:
        """运行问题族独立校验器；仅在可行解通过检查后标记 verified。"""

        session = self._get_internal_session(session_id)
        report = self._reports.get(session_id)
        solver_result = self._results.get(session_id)
        if session.state != "solved" or report is None or solver_result is None:
            raise ValueError("会话尚无可验证的求解结果")
        if report.is_valid is not None:
            return report.model_copy(deep=True)
        if solver_result.status not in {"OPTIMAL", "FEASIBLE"}:
            report.is_valid = False
            report.validation_errors = [
                f"求解状态为 {solver_result.status}，没有可行解可供独立验证。"
            ]
            return report.model_copy(deep=True)

        analysis = session.analysis
        if analysis.problem_family == "employee_scheduling":
            problem = to_scheduling_problem(analysis)
            assignments = [
                Assignment(**item)
                for item in report.result["assignments"]
            ]
            validation = validate_solution(problem, assignments)
        elif analysis.problem_family == "linear_programming":
            problem = to_linear_program_problem(analysis)
            validation = validate_linear_solution(
                problem,
                dict(solver_result.variable_values),
                solver_result.objective_value,
            )
        else:
            problem = to_minimum_cost_flow_problem(analysis)
            arc_flows = decode_flow_solution(
                problem,
                solver_result.variable_values,
            )
            if not isinstance(solver_result.objective_value, int):
                raise ValueError("网络流总费用必须为整数")
            validation = validate_min_cost_flow_solution(
                problem,
                arc_flows,
                solver_result.objective_value,
            )

        report.is_valid = validation.is_valid
        report.validation_errors = list(validation.errors)
        session.state = "verified"
        return report.model_copy(deep=True)
