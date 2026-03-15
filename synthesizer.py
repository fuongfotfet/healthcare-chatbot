import json
import re
from langchain_openai import ChatOpenAI
import config
from schemas import RouterState
from prompts import SYNTHESIZER_PROMPT


class GlobalSynthesizerNode:
    """Class tổng hợp dữ liệu, tích hợp cơ chế Smart Bypass để tối ưu chi phí"""

    def __init__(self):
        print("⏳ [Synthesizer] Initializing Global Synthesizer...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1)

    def process(self, state: RouterState):
        query = state["query"]
        reports = state.get("specialty_reports", {})
        source_mapping = state.get("source_mapping", {})

        if not reports:
            return {
                "response": "Tôi chưa đủ thông tin để định vị chính xác vấn đề của bạn. Hãy mô tả thêm chi tiết nhé."}

        if len(reports) == 1:
            domain_name = list(reports.keys())[0]
            print(f"⚡ [Global Synthesizer] Bệnh chỉ thuộc 1 khoa ({domain_name.upper()}). Bỏ qua bước tổng hợp LLM để tối ưu tốc độ!")

            # Lấy thẳng bài viết của Đặc vụ làm kết quả cuối cùng
            final_answer = list(reports.values())[0]

        else:
            print(f"🧬 [Global Synthesizer] Merging {len(reports)} data streams...")
            all_reports_text = ""
            for domain, content in reports.items():
                all_reports_text += f"\n--- BÁO CÁO TỪ KHOA {domain.upper()} ---\n{content}\n"

            prompt = SYNTHESIZER_PROMPT.format(
                all_reports_text=all_reports_text,
                query=query
            )
            res = self.llm.invoke(prompt)
            final_answer = res.content

        # ========================================================
        # 🛠️ LOGIC BÓC TÁCH JSON (Dùng chung cho cả 2 trường hợp)
        # ========================================================
        pattern = r'<source\s+id="([^"]+)">(.*?)</source>'

        def replace_citation(match):
            raw_id = match.group(1).strip()
            used_text = match.group(2)

            ref_id = raw_id if raw_id.startswith("[") else f"[{raw_id}]"

            meta = source_mapping.get(ref_id)
            if meta:
                single_source_data = {
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "chunk_text": meta.get("chunk_text", "").replace("\n", " "),
                    "used_text": used_text
                }
                inline_json = f"\n```json\n{json.dumps(single_source_data, ensure_ascii=False, indent=2)}\n```\n"
                return f"{used_text} {inline_json}"

            return used_text

        final_answer = re.sub(pattern, replace_citation, final_answer, flags=re.DOTALL)

        return {"response": final_answer}