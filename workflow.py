from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from schemas import RouterState

from llmrouting import IntentAnalyzerNode
from retrieval import VectorRetrievalNode
from experts import DomainExpertsNode
from synthesizer import GlobalSynthesizerNode


class MedicalWorkflow:
    def __init__(self):
        self.router = IntentAnalyzerNode()
        self.retriever = VectorRetrievalNode()
        self.experts = DomainExpertsNode()
        self.synthesizer = GlobalSynthesizerNode()

        self.memory = MemorySaver()
        self.app = self._build_graph()

    def route_logic(self, state: RouterState) -> str:
        if not state.get("analyzed_specialties"):
            return "synthesis_node"
        return "vector_retrieval_node"

    # ĐÃ XÓA HÀM route_after_experts GÂY LỖI ĐI ĐƯỜNG TẮT

    def _build_graph(self):
        builder = StateGraph(RouterState)

        builder.add_node("intent_analyzer", self.router.process)
        builder.add_node("vector_retrieval", self.retriever.process)
        builder.add_node("domain_experts", self.experts.process)
        builder.add_node("global_synthesizer", self.synthesizer.process)

        builder.add_edge(START, "intent_analyzer")
        builder.add_conditional_edges(
            "intent_analyzer", self.route_logic,
            {"vector_retrieval_node": "vector_retrieval", "synthesis_node": "global_synthesizer"}
        )
        builder.add_edge("vector_retrieval", "domain_experts")
        builder.add_edge("domain_experts", "global_synthesizer")

        builder.add_edge("global_synthesizer", END)

        return builder.compile(
            checkpointer=self.memory,
            interrupt_before=["vector_retrieval"]
        )