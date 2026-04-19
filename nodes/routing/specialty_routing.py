from langchain_openai import ChatOpenAI
import re
import unicodedata
from core import config
from core.schemas import RouterState
from core.database import DatabaseManager


class SpecialtyRoutingNode:
    """Class đảm nhiệm việc phân tích ý định và định tuyến (Routing)"""

    FORCED_SPECIALTY_ALIAS = "nam_khoa"

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

    @staticmethod
    def _normalize_specialty_alias(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text or "")
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = normalized.lower().strip().replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"_+", "_", normalized)
        return normalized

    def _resolve_forced_specialty_name(self, valid_domains: list[str]) -> str:
        for domain in valid_domains:
            if self._normalize_specialty_alias(domain) == self.FORCED_SPECIALTY_ALIAS:
                return domain
        return self.FORCED_SPECIALTY_ALIAS

    def process(self, state: RouterState):
        query = state["query"]

        valid_domains = self._load_valid_domains()

        if not valid_domains:
            print("⚠️ [Router] Không tìm thấy chuyên khoa hợp lệ trong bảng guidelines.")
            return {"analyzed_specialties": [], "hypothetical_document": ""}

        forced_specialty = self._resolve_forced_specialty_name(valid_domains)
        print(f"🧭 [Router] Điều phối chuyên khoa ưu tiên: {forced_specialty}")
        return {
            "analyzed_specialties": [{"name": forced_specialty}],
            "hypothetical_document": query,
        }
