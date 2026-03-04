from langgraph.graph import StateGraph, START, END
from schemas import RouterState

# Import các Objects từ các file riêng biệt
from llmrouting import IntentAnalyzerNode
from retrieval import VectorRetrievalNode
from experts import DomainExpertsNode
from synthesizer import GlobalSynthesizerNode


class MedicalWorkflow:
    """Bảng mạch chủ (Orchestrator) kết nối các Nodes thành một luồng LangGraph"""

    def __init__(self):
        self.router = IntentAnalyzerNode()
        self.retriever = VectorRetrievalNode()
        self.experts = DomainExpertsNode()
        self.synthesizer = GlobalSynthesizerNode()
        self.app = self._build_graph()

    # --- Edge Logic (Điều kiện rẽ nhánh) ---
    def route_logic(self, state: RouterState) -> str:
        if not state["analyzed_specialties"]:
            return "synthesis_node"
        return "vector_retrieval_node"

    def route_after_experts(self, state: RouterState) -> str:
        if len(state.get("analyzed_specialties", [])) <= 1:
            print("🚀 [Bypass Optimizer] Single-domain detected. Bypassing Synthesizer!")
            return "end"
        return "synthesis_node"

    # --- Khâu nối các Module ---
    def _build_graph(self):
        builder = StateGraph(RouterState)  # type: ignore

        builder.add_node("intent_analyzer", self.router.process)  # type: ignore
        builder.add_node("vector_retrieval", self.retriever.process)  # type: ignore
        builder.add_node("domain_experts", self.experts.process)  # type: ignore
        builder.add_node("global_synthesizer", self.synthesizer.process)  # type: ignore

        builder.add_edge(START, "intent_analyzer")
        builder.add_conditional_edges(
            "intent_analyzer", self.route_logic,
            {"vector_retrieval_node": "vector_retrieval", "synthesis_node": "global_synthesizer"}
        )
        builder.add_edge("vector_retrieval", "domain_experts")
        builder.add_conditional_edges(
            "domain_experts", self.route_after_experts,
            {"end": END, "synthesis_node": "global_synthesizer"}
        )
        builder.add_edge("global_synthesizer", END)

        return builder.compile()