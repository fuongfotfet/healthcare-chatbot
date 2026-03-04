import asyncio
from workflow import MedicalWorkflow


class ChatbotApp:
    def __init__(self):
        # Khởi tạo toàn bộ luồng từ file workflow
        self.workflow = MedicalWorkflow()
        self.app = self.workflow.app

    async def run_chat_loop(self):
        print("\n" + "=" * 80)
        print("💻 MEDICAL RAG SYSTEM (OOP ARCHITECTURE)")
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