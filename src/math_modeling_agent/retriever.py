"""基于 HMML 层级关键词检索候选建模方法。"""

from dataclasses import dataclass

from .hmml import HMMLDomain, HMMLLibrary, HMMLMethod, HMMLSubdomain, load_hmml


@dataclass(frozen=True)
class MethodRecommendation:
    """一个方法候选及其来源、适用信息和相关度分数。"""

    domain_id: str
    domain_name: str
    subdomain_id: str
    subdomain_name: str
    method_id: str
    method_name: str
    core_idea: str
    use_when: list[str]
    assumptions: list[str]
    limitations: list[str]
    implementation_status: str
    solver: str | None
    problem_score: float
    goal_score: float
    relevance_score: float


def prioritize_compatible_linear_methods(
    recommendations: list[MethodRecommendation],
    *,
    has_discrete_variables: bool,
) -> list[MethodRecommendation]:
    """把与变量域匹配的线性求解方法置顶并移除不兼容后端。"""

    preferred_method_id = (
        "integer_programming"
        if has_discrete_variables
        else "continuous_linear_programming"
    )
    incompatible_method_id = (
        "continuous_linear_programming"
        if has_discrete_variables
        else "integer_programming"
    )
    compatible = [
        method
        for method in recommendations
        if method.method_id != incompatible_method_id
    ]
    preferred = next(
        (method for method in compatible if method.method_id == preferred_method_id),
        None,
    )
    if preferred is None:
        return compatible
    return [
        preferred,
        *(method for method in compatible if method.method_id != preferred_method_id),
    ]


def _keyword_hits(text: str, keywords: list[str]) -> int:
    """统计知识卡片关键词在查询文本中的命中数。"""

    normalized_text = text.casefold()
    return sum(
        1
        for keyword in set(keywords)
        if keyword and keyword.casefold() in normalized_text
    )


class HMMLRetriever:
    """沿领域、子领域、方法三层检索并排序候选方法。"""

    def __init__(self, library: HMMLLibrary | None = None) -> None:
        self.library = library or load_hmml()

    def retrieve(
        self,
        problem_description: str,
        desired_outcome: str = "",
        top_k: int = 3,
    ) -> list[MethodRecommendation]:
        """分别匹配问题描述和目标，再合并为排序分数。"""

        if top_k <= 0:
            raise ValueError("top_k 必须是正整数")

        if not problem_description.strip() and not desired_outcome.strip():
            return []

        candidates: list[MethodRecommendation] = []
        for domain in self.library.domains:
            for subdomain in domain.subdomains:
                for method in subdomain.methods:
                    problem_score = self._hierarchy_score(
                        problem_description,
                        domain,
                        subdomain,
                        method,
                        "problem_keywords",
                    )
                    goal_score = self._hierarchy_score(
                        desired_outcome,
                        domain,
                        subdomain,
                        method,
                        "goal_keywords",
                    )

                    relevance_score = 0.7 * problem_score + 0.3 * goal_score
                    if relevance_score <= 0:
                        continue

                    candidates.append(
                        MethodRecommendation(
                            domain_id=domain.id,
                            domain_name=domain.name,
                            subdomain_id=subdomain.id,
                            subdomain_name=subdomain.name,
                            method_id=method.id,
                            method_name=method.name,
                            core_idea=method.core_idea,
                            use_when=method.use_when,
                            assumptions=method.assumptions,
                            limitations=method.limitations,
                            implementation_status=method.implementation_status,
                            solver=method.solver,
                            problem_score=round(problem_score, 3),
                            goal_score=round(goal_score, 3),
                            relevance_score=round(relevance_score, 3),
                        )
                    )

        candidates.sort(
            key=lambda candidate: (
                -candidate.relevance_score,
                candidate.method_id,
            )
        )
        return candidates[:top_k]

    @staticmethod
    def _hierarchy_score(
        text: str,
        domain: HMMLDomain,
        subdomain: HMMLSubdomain,
        method: HMMLMethod,
        keyword_field: str,
    ) -> float:
        """让命中越具体层级的关键词贡献越高。"""

        if not text.strip():
            return 0.0

        domain_keywords = getattr(domain, keyword_field)
        subdomain_keywords = getattr(subdomain, keyword_field)
        method_keywords = getattr(method, keyword_field)
        if keyword_field == "problem_keywords":
            method_keywords = [*method_keywords, method.name, *method.aliases]

        return (
            1.0 * _keyword_hits(text, domain_keywords)
            + 2.0 * _keyword_hits(text, subdomain_keywords)
            + 3.0 * _keyword_hits(text, method_keywords)
        )
