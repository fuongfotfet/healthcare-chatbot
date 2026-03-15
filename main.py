import asyncio
import os
from dotenv import load_dotenv

# 🚀 ÉP TẢI BIẾN MÔI TRƯỜNG NGAY TỪ DÒNG ĐẦU TIÊN (Rất quan trọng!)
# Phải load xong chìa khóa thì mới được phép import các module bên dưới
load_dotenv(override=True)

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

        # Khởi tạo ID phiên làm việc cho bộ nhớ của LangGraph (Bắt buộc)
        run_config = {"configurable": {"thread_id": "terminal_session_1"}}

        while True:
            q = input("\n> [User Input] (gõ 'exit' để thoát): ")
            if q.lower() == 'exit':
                break

            try:
                # Chạy luồng đồ thị (Kèm theo config bộ nhớ)
                result = await self.app.ainvoke({"query": q}, config=run_config)
                print(f"\n[System Output]: \n{result.get('response', '')}")
            except Exception as e:
                print(f"\n❌ [Lỗi hệ thống]: {e}")


if __name__ == "__main__":
    chatbot = ChatbotApp()
    asyncio.run(chatbot.run_chat_loop())