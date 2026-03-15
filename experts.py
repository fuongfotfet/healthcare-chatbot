import asyncio
from langchain_openai import ChatOpenAI
import config
from schemas import RouterState
from prompts import EXPERT_PROMPT

class DomainExpertsNode:
    def __init__(self):
        print("⏳ [Experts] Initializing Expert Agents...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1)
    async def process(self, state: RouterState):
        query = state["query"]
        contexts = state.get("specialty_contexts", {})
        analyzed_domains = state.get("analyzed_specialties", [])
        core_map = {spec["name"]: spec["is_core_issue"] for spec in analyzed_domains}

        async def generate_single_report(domain_name, context):
            is_core = core_map.get(domain_name, False)
            role = "🔴 [CORE_ISSUE]" if is_core else "⚪ [MINOR_ISSUE]"

            prompt = EXPERT_PROMPT.format(
                domain_name=domain_name.upper(),
                role=role,
                context=context,
                query=query
            )

            res = await self.llm.ainvoke(prompt)
            return domain_name, res.content

        tasks = [generate_single_report(name, ctx) for name, ctx in contexts.items()]
        results = await asyncio.gather(*tasks) if tasks else []
        reports = {name: content for name, content in results}

        # CHỈ TRẢ VỀ REPORTS ĐỂ TRƯỞNG KHOA LÀM VIỆC TIẾP
        return {"specialty_reports": reports}