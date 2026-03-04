import asyncio
from langgraph.graph import StateGraph, START, END
from schemas import RouterState
from agents import MedicalWorkflowManager


class ChatbotApp:
    """Class khởi chạy và cấu hình Graph LangChain"""

    def __init__(self):
        self.workflow = MedicalWorkflowManager()
        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RouterState)  # type: ignore

        # Đã cập nhật lại tên các node theo chuẩn kỹ thuật
        builder.add_node("intent_analyzer", self.workflow.intent_analyzer_node)  # type: ignore
        builder.add_node("vector_retrieval", self.workflow.vector_retrieval_node)  # type: ignore
        builder.add_node("domain_experts", self.workflow.domain_expert_node)  # type: ignore
        builder.add_node("global_synthesizer", self.workflow.synthesis_node)  # type: ignore

        # Thêm các luồng (Edges)
        builder.add_edge(START, "intent_analyzer")
        builder.add_conditional_edges(
            "intent_analyzer",
            self.workflow.route_logic,
            {"vector_retrieval_node": "vector_retrieval", "synthesis_node": "global_synthesizer"}
        )
        builder.add_edge("vector_retrieval", "domain_experts")
        builder.add_conditional_edges(
            "domain_experts",
            self.workflow.route_after_experts,
            {"end": END, "synthesis_node": "global_synthesizer"}
        )
        builder.add_edge("global_synthesizer", END)

        return builder.compile()

    async def run_chat_loop(self):
        print("\n" + "=" * 80)
        print("💻 MEDICAL RAG SYSTEM (MULTI-AGENT ORCHESTRATOR)")
        print("=" * 80)

        while True:
            q = input("\n> [User Input] (gõ 'exit' để thoát): ")
            if q.lower() == 'exit':
                break

            result = await self.app.ainvoke({"query": q})
            print(f"\n[System Output]: \n{result.get('response', '')}")


if __name__ == "__main__":
    chatbot = ChatbotApp()
    asyncio.run(chatbot.run_chat_loop())