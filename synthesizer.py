from langchain_google_genai import ChatGoogleGenerativeAI
import config
from schemas import RouterState


class GlobalSynthesizerNode:
    """Class tổng hợp các luồng dữ liệu thành phản hồi cuối cùng"""

    def __init__(self):
        print("⏳ [Synthesizer] Initializing Global Synthesizer...")
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2)

    def process(self, state: RouterState):
        query = state["query"]
        reports = state.get("specialty_reports", {})
        analyzed_domains = state.get("analyzed_specialties", [])

        if not reports:
            return {"response": "System Fault: Không thể ánh xạ dữ liệu."}

        print(f"🧬 [Global Synthesizer] Merging {len(reports)} data streams...")
        core_issue_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_domains}

        all_reports_text = ""
        for domain, content in reports.items():
            priority = "[CORE_ISSUE]" if core_issue_map.get(domain, False) else "[MINOR_ISSUE]"
            all_reports_text += f"\n--- DATA STREAM: {domain.upper()} {priority} ---\n{content}\n"

        prompt = f"""Bạn là Global Synthesis Agent. Tổng hợp các luồng dữ liệu thành một báo cáo duy nhất, mượt mà.
        [DATA STREAMS]:\n{all_reports_text}\n\n[USER INPUT]: {query}
        Yêu cầu: Tập trung 80% vào CORE_ISSUE, giữ lại Trích dẫn (Source), thêm Disclaimer cuối cùng."""

        res = self.llm.invoke(prompt)
        return {"response": res.content}