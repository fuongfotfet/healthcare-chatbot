from langchain_google_genai import ChatGoogleGenerativeAI
import config
from schemas import RouterState, RouteDecision


class IntentAnalyzerNode:
    """Class đảm nhiệm việc phân tích ý định và định tuyến (Routing)"""

    def __init__(self):
        print("⏳ [Router] Initializing Intent Analyzer...")
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0)

    def process(self, state: RouterState):
        query = state["query"]
        structured_llm = self.llm.with_structured_output(RouteDecision)

        prompt = f"Phân tích input người dùng, phân loại domain và tạo query mở rộng (HyDE).\nInput: {query}"
        decision = structured_llm.invoke(prompt)

        # Danh sách ĐẦY ĐỦ 22 chuyên khoa đã khớp với Database
        valid_domains = [
            "tim_mach", "ho_hap", "tieu_hoa", "than_kinh", "xuong_khop",
            "da_lieu", "nhi_khoa", "hiv_aids", "huyet_hoc", "phuc_hoi_chuc_nang",
            "duoc_hoc", "tam_than", "noi_tiet", "ung_buou", "ky_sinh_trung",
            "nhiem_khuan", "dinh_duong", "y_te_cong_cong", "truyen_nhiem",
            "san_phu_khoa", "hau_covid", "cap_cuu"
        ]

        filtered_domains = [{"name": s.name, "is_core_issue": s.is_core_issue}
                            for s in decision.analyzed_specialties if s.name in valid_domains]

        print(f"🧭 [Router] Điều phối đến các domain: {[s['name'] for s in filtered_domains]}")
        return {"analyzed_specialties": filtered_domains, "hypothetical_document": decision.hypothetical_document}