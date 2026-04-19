import streamlit as st
import asyncio
from app.cli import ChatbotApp

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Trợ lý Y khoa Đa đặc vụ", page_icon="🏥", layout="centered")
st.title("🏥 Trợ lý Y khoa Đa đặc vụ")

@st.cache_resource
def load_ai_engine():
    return ChatbotApp()

chatbot_engine = load_ai_engine()


def format_runtime_error(exc):
    raw = str(exc)
    lowered = raw.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            "❌ Tài khoản OpenAI hiện không đủ quota. "
            "Vui lòng nạp quota hoặc đổi sang API key/project khác còn hạn mức."
        )
    if "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return "❌ OPENAI_API_KEY không hợp lệ. Vui lòng kiểm tra lại biến môi trường hoặc file .env."
    if "rate limit" in lowered or "429" in lowered:
        return "❌ OpenAI đang giới hạn tần suất gọi API. Vui lòng đợi một chút rồi thử lại."
    return f"❌ Đã xảy ra lỗi: {raw}"


def run_async(coro):
    """Run async coroutine safely in Streamlit context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# --- 2. KHỞI TẠO BỘ NHỚ ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Xin chào! Hãy mô tả triệu chứng, tôi sẽ giúp bạn định hướng chuyên khoa nhé."}
    ]

# Hiển thị lại toàn bộ lịch sử
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. NHẬP CÂU HỎI MỚI ---
if user_query := st.chat_input("Ví dụ: Tôi hay bị tê mỏi ngón tay và đau khớp gối..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        trace_box = st.expander("🧠 Quá trình suy luận", expanded=False)
        trace_placeholder = trace_box.empty()
        final_answer_parts = []
        trace_parts = []

        async def stream_to_ui():
            async for event, payload in chatbot_engine.stream_answer_events(user_query):
                if not payload:
                    continue
                if event == "trace":
                    trace_parts.append(f"- {payload}")
                    trace_placeholder.markdown("\n".join(trace_parts))
                    continue
                final_answer_parts.append(payload)
                placeholder.markdown("".join(final_answer_parts))

        try:
            run_async(stream_to_ui())
            final_answer = "".join(final_answer_parts)
        except Exception as e:
            final_answer = format_runtime_error(e)
            placeholder.error(final_answer)

        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# --- 4. NÚT XÓA LỊCH SỬ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử hội chẩn"):
        st.session_state.chat_history = []
        st.rerun()