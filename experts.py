import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
import config
from schemas import RouterState


class DomainExpertsNode:
    """Class quản lý các Đặc vụ AI xử lý độc lập từng chuyên khoa"""

    def __init__(self):
        print("⏳ [Experts] Initializing Expert Agents...")
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.1)

    async def process(self, state: RouterState):
        query = state["query"]
        contexts = state.get("specialty_contexts", {})
        analyzed_domains = state.get("analyzed_specialties", [])
        core_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_domains}

        async def generate_single_report(domain_name, context):
            is_core = core_map.get(domain_name, False)
            role = "🔴 [CORE_ISSUE] (Cốt lõi): Yêu cầu phân tích chi tiết và chuyên sâu." if is_core else "⚪ [MINOR_ISSUE] (Thứ cấp): Yêu cầu phân tích chi tiết đầy đủ, không được tóm tắt."

            print(f"⚙️ [Expert - {domain_name.upper()}] Processing...")

            prompt = f"""Bạn là AI Domain Expert quản lý phân hệ {domain_name.upper()}. {role}
            [CONTEXT DATA]:\n{context}\n\n[USER INPUT]: {query}"""

            res = await self.llm.ainvoke(prompt)
            return domain_name, res.content

        tasks = [generate_single_report(name, ctx) for name, ctx in contexts.items()]
        results = await asyncio.gather(*tasks) if tasks else []
        reports = {name: content for name, content in results}

        if len(reports) == 1:
            single_domain = list(reports.keys())[0]
            return {"specialty_reports": reports,
                    "response": reports[single_domain] + "\n\n⚠️ Disclaimer: AI chỉ mang tính tham khảo."}

        return {"specialty_reports": reports}