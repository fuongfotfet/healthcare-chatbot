from langchain_openai import ChatOpenAI
from core import config
from core.schemas import RouterState, RouteDecision
from core.prompts import ROUTER_PROMPT
from core.database import DatabaseManager


class SpecialtyRoutingNode:
    """Class đảm nhiệm việc phân tích ý định và định tuyến (Routing)"""

    def __init__(self):
        print("⏳ [Router] Initializing Intent Analyzer...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)
        self.db_manager = DatabaseManager()

    def _load_valid_domains(self):
        """Lấy danh sách chuyên khoa từ bảng guidelines."""
        conn = None
        cursor = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT chuyen_khoa
                FROM guidelines
                WHERE chuyen_khoa IS NOT NULL
                  AND btrim(chuyen_khoa) <> ''
                ORDER BY chuyen_khoa;
                """
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"❌ [Router DB Error] Không tải được valid domains: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def process(self, state: RouterState):
        query = state["query"]
        structured_llm = self.llm.with_structured_output(RouteDecision)

        valid_domains = self._load_valid_domains()

        if not valid_domains:
            print("⚠️ [Router] Không tìm thấy chuyên khoa hợp lệ trong bảng guidelines.")
            return {"analyzed_specialties": [], "hypothetical_document": ""}

        # Biến danh sách trên thành một chuỗi văn bản (VD: "tim_mach, ho_hap, ...")
        domains_string = ", ".join(valid_domains)

        # GỌI PROMPT TỪ FILE MỚI VÀ TRUYỀN BIẾN VÀO
        prompt = ROUTER_PROMPT.format(
            domains_string=domains_string,
            query=query
        )

        decision = structured_llm.invoke(prompt)

        filtered_domains = [{"name": s.name}
                            for s in decision.analyzed_specialties if s.name in valid_domains]

        print(f"🧭 [Router] Điều phối đến các domain: {[s['name'] for s in filtered_domains]}")
        return {"analyzed_specialties": filtered_domains, "hypothetical_document": decision.hypothetical_document}
