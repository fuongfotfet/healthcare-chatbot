import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import config
from schemas import RouterState, RouteDecision
from database import DatabaseManager


class MedicalWorkflowManager:
    """Class quản lý các Node (Đặc vụ) và luồng xử lý Multi-Agent RAG"""

    def __init__(self):
        print("⏳ [System] Initializing LLM and Embedding models...")
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.db_manager = DatabaseManager()

    def intent_analyzer_node(self, state: RouterState):
        """Node 1: Phân tích ý định người dùng và định tuyến (Routing)"""
        query = state["query"]
        structured_llm = self.llm.with_structured_output(RouteDecision)

        prompt = f"Phân tích input người dùng, phân loại domain và tạo query mở rộng (HyDE).\nInput: {query}"
        decision = structured_llm.invoke(prompt)

        valid_domains = ["tim_mach", "ho_hap", "tieu_hoa", "than_kinh", "xuong_khop", "da_lieu", "nhi_khoa"]
        filtered_domains = [{"name": s.name, "is_core_issue": s.is_core_issue} for s in decision.analyzed_specialties if
                            s.name in valid_domains]

        print(f"🧭 [Router] Luồng dữ liệu được điều phối đến các domain: {[s['name'] for s in filtered_domains]}")
        return {"analyzed_specialties": filtered_domains, "hypothetical_document": decision.hypothetical_document}

    async def vector_retrieval_node(self, state: RouterState):
        """Node 2: Truy xuất song song từ Vector Database"""
        hyde_text = state["hypothetical_document"]
        domains_data = state["analyzed_specialties"]

        print(f"🔍 [Retriever] Đang thực thi truy vấn song song (Async) trên {len(domains_data)} domain...")
        query_vec = self.embed_model.encode(hyde_text).tolist()

        def fetch_from_db(domain_item):
            domain_name = domain_item["name"]
            chunk_limit = 4 if domain_item["is_core_issue"] else 2

            try:
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT d.title, d.source, c.content 
                    FROM document_chunks c JOIN medical_documents d ON c.document_id = d.id
                    WHERE d.specialty = %s ORDER BY c.embedding <=> %s::vector LIMIT %s;
                """, (domain_name, query_vec, chunk_limit))

                results = cursor.fetchall()
                cursor.close()
                conn.close()

                if results:
                    chunks = [f"📌 Source: {row[0]}\n📝 Content: {row[2]}" for row in results]
                    return domain_name, "\n\n".join(chunks)
                else:
                    return domain_name, "No data retrieved."
            except Exception as e:
                print(f"❌ [DB Error] Domain {domain_name}: {e}")
                return domain_name, "System error during retrieval."

        tasks = [asyncio.to_thread(fetch_from_db, spec) for spec in domains_data]
        results = await asyncio.gather(*tasks) if tasks else []

        specialty_contexts = {name: context for name, context in results}
        return {"specialty_contexts": specialty_contexts}

    async def domain_expert_node(self, state: RouterState):
        """Node 3: Các đặc vụ chuyên môn xử lý dữ liệu độc lập (Parallel Processing)"""
        query = state["query"]
        contexts = state["specialty_contexts"]
        analyzed_domains = state.get("analyzed_specialties", [])
        core_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_domains}

        async def generate_single_report(domain_name, context):
            is_core = core_map.get(domain_name, False)
            if is_core:
                print(f"⚙️ [Expert Agent - {domain_name.upper()}] Processing [CORE_ISSUE]...")
                role_awareness = "🔴 THUỘC TÍNH: [CORE_ISSUE] (Vấn đề cốt lõi). Yêu cầu phân tích chuyên sâu, trích xuất phác đồ chi tiết. Flag cảnh báo khẩn cấp nếu khớp với tiêu chuẩn cấp cứu y tế."
            else:
                print(f"⚙️ [Expert Agent - {domain_name.upper()}] Processing [MINOR_ISSUE]...")
                role_awareness = "⚪ THUỘC TÍNH: [MINOR_ISSUE] (Vấn đề thứ cấp). Yêu cầu tóm tắt dữ liệu tối giản (1-2 câu), không làm loãng luồng thông tin chính."

            prompt = f"""Bạn là AI Domain Expert quản lý phân hệ {domain_name.upper()}. 
            {role_awareness}

            CHỈ THỊ: Xử lý input dựa vào khối dữ liệu được cung cấp dưới đây. Bỏ qua các khía cạnh không thuộc phân hệ của bạn.

            [CONTEXT DATA - {domain_name.upper()}]:
            {context}

            [USER INPUT]: {query}
            """
            res = await self.llm.ainvoke(prompt)
            return domain_name, res.content

        tasks = [generate_single_report(name, ctx) for name, ctx in contexts.items()]
        results = await asyncio.gather(*tasks) if tasks else []
        reports = {domain_name: content for domain_name, content in results}

        # Bypass Logic cho trường hợp Single-Domain
        if len(reports) == 1:
            single_domain = list(reports.keys())[0]
            final_response = reports[single_domain] + "\n\n⚠️ System Disclaimer: Phân tích được tạo bởi AI chỉ mang tính chất tham khảo. Vui lòng tham vấn chuyên gia y tế cho các quyết định lâm sàng."
            return {"specialty_reports": reports, "response": final_response}

        return {"specialty_reports": reports}

    def synthesis_node(self, state: RouterState):
        """Node 4: Bộ tổng hợp cuối cùng kết xuất phản hồi (Final Response Generation)"""
        query = state["query"]
        reports = state.get("specialty_reports", {})
        analyzed_domains = state.get("analyzed_specialties", [])

        if not reports:
            return {"response": "System Fault: Không thể ánh xạ dữ liệu phù hợp với truy vấn."}

        print(f"🧬 [Global Synthesizer] Merging {len(reports)} data streams...")
        core_issue_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_domains}

        all_reports_text = ""
        for domain_name, report_content in reports.items():
            is_core = core_issue_map.get(domain_name, False)
            priority_label = "[CORE_ISSUE]" if is_core else "[MINOR_ISSUE]"
            all_reports_text += f"\n--- DATA STREAM: {domain_name.upper()} {priority_label} ---\n{report_content}\n"

        prompt = f"""Bạn là Global Synthesis Agent. Nhiệm vụ của bạn là tổng hợp các luồng dữ liệu chuyên biệt để xuất ra kết quả cuối cùng cho người dùng.

        [DATA STREAMS (Đã gắn nhãn ưu tiên)]:
        {all_reports_text}

        [USER INPUT]: {query}

        CHỈ THỊ KẾT XUẤT (OUTPUT REQUIREMENTS):
        1. RESOURCE ALLOCATION: Dành 80% dung lượng kết quả để xử lý triệt để luồng [CORE_ISSUE]. Các luồng [MINOR_ISSUE] chỉ cần liệt kê ngắn gọn.
        2. CRITICAL ALERT: Nếu bất kỳ luồng dữ liệu nào flag tình trạng khẩn cấp, ĐẨY CẢNH BÁO NÀY LÊN ĐẦU TIÊN (Block-level alert).
        3. SEAMLESS INTEGRATION: Trình bày dữ liệu mượt mà, logic. KHÔNG dùng cú pháp máy móc như "Phân hệ A trả về...". Đóng vai trò là một hệ thống AI đồng nhất.
        4. Duy trì các trích dẫn (Source) để đảm bảo tính minh bạch (Data traceability).
        5. Gắn disclaimer sau vào cuối output: "⚠️ System Disclaimer: Phân tích được tạo bởi AI chỉ mang tính chất tham khảo. Vui lòng tham vấn chuyên gia y tế cho các quyết định lâm sàng."
        """
        res = self.llm.invoke(prompt)
        return {"response": res.content}

    # --- Edge Logic ---
    def route_logic(self, state: RouterState) -> str:
        if not state["analyzed_specialties"]:
            return "synthesis_node"
        return "vector_retrieval_node"

    def route_after_experts(self, state: RouterState) -> str:
        if len(state.get("analyzed_specialties", [])) <= 1:
            print("🚀 [Bypass Optimizer] Single-domain detected. Bypassing Global Synthesizer!")
            return "end"
        return "synthesis_node"