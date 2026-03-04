import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import config
from schemas import RouterState, RouteDecision
from database import DatabaseManager


class MedicalWorkflowManager:
    """Class quản lý các Node AI và Logic hội chẩn Đa đặc vụ"""

    def __init__(self):
        print("⏳ Đang khởi tạo mô hình AI và Embedding...")
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.db_manager = DatabaseManager()

    def router_node_llm(self, state: RouterState):
        query = state["query"]
        structured_llm = self.llm.with_structured_output(RouteDecision)

        prompt = f"Phân tích triệu chứng, chọn chuyên khoa và viết HyDE.\nTriệu chứng: {query}"
        decision = structured_llm.invoke(prompt)

        valid_specs = ["tim_mach", "ho_hap", "tieu_hoa", "than_kinh", "xuong_khop"]
        filtered_specs = [{"name": s.name, "is_core_issue": s.is_core_issue} for s in decision.analyzed_specialties if
                          s.name in valid_specs]

        print(f"🧭 [Phân tuyến]: Chuyển bệnh án tới các khoa -> {[s['name'] for s in filtered_specs]}")
        return {"analyzed_specialties": filtered_specs, "hypothetical_document": decision.hypothetical_document}

    async def retrieve_node(self, state: RouterState):
        hyde_text = state["hypothetical_document"]
        specialties_data = state["analyzed_specialties"]

        print(f"🔍 [Tra cứu DB]: Đang mở kết nối song song tìm kiếm {len(specialties_data)} khoa...")
        query_vec = self.embed_model.encode(hyde_text).tolist()

        def fetch_from_db(spec_item):
            spec_name = spec_item["name"]
            chunk_limit = 4 if spec_item["is_core_issue"] else 2

            try:
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.title, d.source, c.content 
                    FROM document_chunks c JOIN medical_documents d ON c.document_id = d.id
                    WHERE d.specialty = %s ORDER BY c.embedding <=> %s::vector LIMIT %s;
                """, (spec_name, query_vec, chunk_limit))

                results = cursor.fetchall()
                cursor.close()
                conn.close()

                if results:
                    chunks = [f"📌 Nguồn: {row[0]}\n📝 Nội dung: {row[2]}" for row in results]
                    return spec_name, "\n\n".join(chunks)
                else:
                    return spec_name, "Không tìm thấy dữ liệu."
            except Exception as e:
                print(f"❌ Lỗi DB ở khoa {spec_name}: {e}")
                return spec_name, "Hệ thống tra cứu đang gặp sự cố."

        tasks = [asyncio.to_thread(fetch_from_db, spec) for spec in specialties_data]
        results = await asyncio.gather(*tasks) if tasks else []

        specialty_contexts = {name: context for name, context in results}
        return {"specialty_contexts": specialty_contexts}

    async def specialty_experts_node(self, state: RouterState):
        query = state["query"]
        contexts = state["specialty_contexts"]
        analyzed_specialties = state.get("analyzed_specialties", [])
        core_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_specialties}

        async def generate_single_report(spec_name, context):
            is_core = core_map.get(spec_name, False)
            if is_core:
                print(f"👨‍⚕️ [Bác sĩ {spec_name.upper()}]: Nhận ca BỆNH CHÍNH -> Đang viết báo cáo...")
                role_awareness = "🔴 ĐÂY LÀ VẤN ĐỀ CHÍNH VÀ CẤP BÁCH. Phân tích thật sâu, nếu cần thì yêu cầu gọi 115."
            else:
                print(f"👨‍⚕️ [Bác sĩ {spec_name.upper()}]: Nhận ca BỆNH PHỤ -> Đang viết tóm tắt...")
                role_awareness = "⚪ đây chỉ là triệu chứng phụ. Hãy viết báo cáo THẬT NGẮN GỌN (1-2 câu)."

            prompt = f"""Bạn là Bác sĩ chuyên khoa {spec_name.upper()}. 
            {role_awareness}
            Giải quyết triệu chứng CHỈ DỰA TRÊN TÀI LIỆU CỦA KHOA BẠN.
            TÀI LIỆU KHOA {spec_name.upper()}:
            {context}

            BỆNH NHÂN KHAI: {query}
            """
            res = await self.llm.ainvoke(prompt)
            return spec_name, res.content

        tasks = [generate_single_report(name, ctx) for name, ctx in contexts.items()]
        results = await asyncio.gather(*tasks) if tasks else []
        reports = {spec_name: content for spec_name, content in results}

        # Bypass
        if len(reports) == 1:
            single_spec = list(reports.keys())[0]
            final_response = reports[single_spec] + "\n\n⚠️ Lưu ý: Thông tin trên chỉ mang tính chất tham khảo. Vui lòng đến cơ sở y tế để được thăm khám chính xác."
            return {"specialty_reports": reports, "response": final_response}

        return {"specialty_reports": reports}

    def chief_doctor_node(self, state: RouterState):
        query = state["query"]
        reports = state.get("specialty_reports", {})
        analyzed_specialties = state.get("analyzed_specialties", [])

        if not reports:
            return {"response": "Xin lỗi, tôi không tìm thấy thông tin phù hợp cho tình trạng của bạn."}

        print(f"🏥 [Bác sĩ Trưởng khoa]: Đang tổng hợp {len(reports)} báo cáo...")
        core_issue_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_specialties}

        all_reports_text = ""
        for spec_name, report_content in reports.items():
            is_core = core_issue_map.get(spec_name, False)
            priority_label = "🔴 VẤN ĐỀ CHÍNH" if is_core else "⚪ VẤN ĐỀ PHỤ"
            all_reports_text += f"\n--- BÁO CÁO TỪ KHOA {spec_name.upper()} ({priority_label}) ---\n{report_content}\n"

        prompt = f"""Bạn là Bác sĩ Trưởng khoa. Nhiệm vụ của bạn là tổng hợp báo cáo chuyên khoa để đưa ra lời khuyên cuối cùng.

        [BÁO CÁO ĐÃ PHÂN CẤP]:
        {all_reports_text}

        [BỆNH NHÂN HỎI]: {query}

        YÊU CẦU:
        1. Tập trung 80% tư vấn cho Vấn đề Chính. Vấn đề Phụ chỉ nhắc ngắn gọn.
        2. Nếu có cảnh báo khẩn cấp, ĐẶT LÊN ĐẦU TIÊN và yêu cầu gọi 115.
        3. KHÔNG liệt kê kiểu "Khoa A nói...". Hãy đóng vai 1 bác sĩ duy nhất.
        4. Giữ lại trích dẫn nguồn.
        5. Thêm câu chốt: "⚠️ Lưu ý: Thông tin chỉ mang tính tham khảo..." ở cuối.
        """
        res = self.llm.invoke(prompt)
        return {"response": res.content}

    # --- Edge Logic ---
    def route_logic(self, state: RouterState) -> str:
        if not state["analyzed_specialties"]:
            return "chief_doctor"
        return "retrieve_db"

    def route_after_experts(self, state: RouterState) -> str:
        if len(state.get("analyzed_specialties", [])) <= 1:
            print("🚀 [Luồng đi tắt]: Chỉ có 1 chuyên khoa, KHÔNG CẦN gọi Trưởng khoa!")
            return "end"
        return "chief_doctor"