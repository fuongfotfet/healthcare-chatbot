from langchain_openai import ChatOpenAI
import config
from schemas import RouterState, RouteDecision
from prompts import ROUTER_PROMPT


class IntentAnalyzerNode:
    """Class đảm nhiệm việc phân tích ý định và định tuyến (Routing)"""

    def __init__(self):
        print("⏳ [Router] Initializing Intent Analyzer...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)

    def process(self, state: RouterState):
        query = state["query"]
        structured_llm = self.llm.with_structured_output(RouteDecision)

        # Danh sách ĐẦY ĐỦ 22 chuyên khoa đã khớp với Database
        valid_domains = [
            "tim_mach", "ho_hap", "tieu_hoa", "than_kinh", "xuong_khop",
            "da_lieu", "nhi_khoa", "hiv_aids", "huyet_hoc", "phuc_hoi_chuc_nang",
            "duoc_hoc", "tam_than", "noi_tiet", "ung_buou", "ky_sinh_trung",
            "nhiem_khuan", "dinh_duong", "y_te_cong_cong", "truyen_nhiem",
            "san_phu_khoa", "hau_covid", "cap_cuu"
        ]

        # Biến danh sách trên thành một chuỗi văn bản (VD: "tim_mach, ho_hap, ...")
        domains_string = ", ".join(valid_domains)

        # GỌI PROMPT TỪ FILE MỚI VÀ TRUYỀN BIẾN VÀO
        prompt = ROUTER_PROMPT.format(
            domains_string=domains_string,
            query=query
        )

        decision = structured_llm.invoke(prompt)

        filtered_domains = [{"name": s.name, "is_core_issue": s.is_core_issue}
                            for s in decision.analyzed_specialties if s.name in valid_domains]

        print(f"🧭 [Router] Điều phối đến các domain: {[s['name'] for s in filtered_domains]}")
        return {"analyzed_specialties": filtered_domains, "hypothetical_document": decision.hypothetical_document}