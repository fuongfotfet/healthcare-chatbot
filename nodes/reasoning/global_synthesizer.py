import re
from langchain_openai import ChatOpenAI
from core import config
from core.schemas import RouterState
from core.prompts import SYNTHESIZER_PROMPT
from nodes.reasoning.citation_transformer import CitationStreamTransformer


class GlobalSynthesizerNode:
    """Class tổng hợp dữ liệu, tích hợp cơ chế Smart Bypass để tối ưu chi phí"""

    def __init__(self):
        print("⏳ [Synthesizer] Initializing Global Synthesizer...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1)

    @staticmethod
    def _stream_tokens(text):
        tokens = re.findall(r"\S+\s*", text or "")
        return tokens if tokens else [text or ""]

    async def stream_process(self, state: RouterState):
        """Stream final response chunks, transforming source tags incrementally."""
        query = state["query"]
        reports = state.get("specialty_reports", {})
        transformer = CitationStreamTransformer()

        if not reports:
            fallback = (
                "### Kết luận sơ bộ\n"
                "- Tôi chưa đủ thông tin để định vị chính xác vấn đề của bạn.\n"
                "\n"
                "### Bạn có thể bổ sung\n"
                "- Triệu chứng chính (đau ở đâu, mức độ, thời gian kéo dài).\n"
                "- Dấu hiệu đi kèm (sốt, nôn, khó thở, phát ban...).\n"
                "- Tiền sử bệnh và thuốc đang dùng (nếu có)."
            )
            for token in self._stream_tokens(fallback):
                delta = transformer.feed(token)
                if delta:
                    yield delta
            tail = transformer.flush()
            if tail:
                yield tail
            return

        print(f"🧬 [Global Synthesizer] Merging {len(reports)} data streams...")
        all_reports_text = ""
        for branch_name, content in reports.items():
            all_reports_text += f"\n--- BÁO CÁO TỪ NHÁNH {branch_name.upper()} ---\n{content}\n"

        prompt = SYNTHESIZER_PROMPT.format(
            all_reports_text=all_reports_text,
            query=query,
        )

        async for chunk in self.llm.astream(prompt):
            chunk_text = getattr(chunk, "content", "") or ""
            if not chunk_text:
                continue
            delta = transformer.feed(chunk_text)
            if delta:
                yield delta

        tail = transformer.flush()
        if tail:
            yield tail

    def process(self, state: RouterState):
        raise RuntimeError(
            "Non-stream mode has been disabled. Use stream_process(...) instead."
        )