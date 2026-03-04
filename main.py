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

        # Thêm các Node
        builder.add_node("router", self.workflow.router_node_llm)  # type: ignore
        builder.add_node("retrieve_db", self.workflow.retrieve_node)  # type: ignore
        builder.add_node("specialty_experts", self.workflow.specialty_experts_node)  # type: ignore
        builder.add_node("chief_doctor", self.workflow.chief_doctor_node)  # type: ignore

        # Thêm các luồng (Edges)
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            self.workflow.route_logic,
            {"retrieve_db": "retrieve_db", "chief_doctor": "chief_doctor"}
        )
        builder.add_edge("retrieve_db", "specialty_experts")
        builder.add_conditional_edges(
            "specialty_experts",
            self.workflow.route_after_experts,
            {"end": END, "chief_doctor": "chief_doctor"}
        )
        builder.add_edge("chief_doctor", END)

        return builder.compile()

    async def run_chat_loop(self):
        print("\n" + "=" * 80)
        print("🏥 HỆ THỐNG RAG ĐA ĐẶC VỤ (OOP MODULAR STRUCTURE)")
        print("=" * 80)

        while True:
            q = input("\n❓ Bệnh nhân hỏi (gõ 'exit' để thoát): ")
            if q.lower() == 'exit':
                break

            result = await self.app.ainvoke({"query": q})
            print(f"\n🩺 KẾT LUẬN TƯ VẤN: \n{result.get('response', '')}")


if __name__ == "__main__":
    chatbot = ChatbotApp()
    asyncio.run(chatbot.run_chat_loop())