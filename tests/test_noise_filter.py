from core.application.indexing.noise_filter import LlmNoiseFilter
from core.application.ports.llm import LlmPort
from core.domain.models import Document, DocumentNode, Section


class NoiseFilterLlm(LlmPort):
    def filter_noise(self, *, sections: list[Section]) -> set[str]:
        return {section.node_id for section in sections if "appendix" in section.text.lower()}

    def rerank(self, *, question: str, candidates: list[Section], k_final: int) -> list[str]:
        del question, candidates, k_final
        return []

    def synthesize(self, *, question: str, sections: list[Section]) -> str:
        del question, sections
        return ""

    def synthesize_stream(self, *, question: str, sections: list[Section]):
        yield self.synthesize(question=question, sections=sections)


def test_llm_noise_filter_removes_noisy_nodes() -> None:
    document = Document(
        doc_id="manual",
        nodes=(
            DocumentNode(
                doc_id="manual",
                node_id="manual:n1",
                level=1,
                order=1,
                title="Intro",
                content="Welcome",
                parent_id=None,
                breadcrumb=("Intro",),
            ),
            DocumentNode(
                doc_id="manual",
                node_id="manual:n2",
                level=1,
                order=2,
                title="Appendix",
                content="Legal boilerplate",
                parent_id=None,
                breadcrumb=("Appendix",),
            ),
        ),
    )

    filtered = LlmNoiseFilter(llm=NoiseFilterLlm()).filter(document)

    assert [node.node_id for node in filtered.nodes] == ["manual:n1"]
