import streamlit as st
import asyncio
from main import ChatbotApp

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Trợ lý Y khoa Đa đặc vụ",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Trợ lý Y khoa Đa đặc vụ")
st.caption("Hội chẩn đồng thời 5 chuyên khoa: Tim mạch, Hô hấp, Tiêu hóa, Thần kinh, Xương khớp.")


# --- 2. KHỞI TẠO BỘ NÃO AI (Chỉ tải 1 lần) ---
# Dùng @st.cache_resource để Streamlit không phải tải lại mô hình AI mỗi lần gõ phím
@st.cache_resource
def load_ai_engine():
    return ChatbotApp()


chatbot_engine = load_ai_engine()

# --- 3. KHỞI TẠO BỘ NHỚ LƯU TRỮ LỊCH SỬ CHAT ---
if "chat_history" not in st.session_state:
    # Nếu chưa có lịch sử, khởi tạo mảng rỗng và hiển thị câu chào
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Xin chào! Tôi là Trợ lý Y khoa. Tôi có thể giúp gì cho bạn?"}
    ]

# Hiển thị lại toàn bộ lịch sử tin nhắn cũ ra màn hình
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. XỬ LÝ KHI BỆNH NHÂN NHẬP CÂU HỎI ---
# Tạo thanh input nhập tin nhắn ở dưới cùng màn hình
if user_query := st.chat_input("Ví dụ: Tôi bị đau thắt ngực trái và vã mồ hôi..."):

    # 4.1. In câu hỏi của người dùng ra màn hình
    with st.chat_message("user"):
        st.markdown(user_query)
    # Lưu vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # 4.2. Khởi động hệ thống Đa đặc vụ (Graph LangChain)
    with st.chat_message("assistant"):
        # Tạo hiệu ứng "Đang suy nghĩ" quay vòng tròn
        with st.spinner("👨‍⚕️ Các bác sĩ chuyên khoa đang hội chẩn..."):
            try:
                # Ép thư viện Streamlit (đồng bộ) chạy hàm LangGraph (bất đồng bộ)
                result = asyncio.run(chatbot_engine.app.ainvoke({"query": user_query}))
                final_answer = result.get('response', 'Hệ thống AI không trả về kết quả.')
            except Exception as e:
                final_answer = f"❌ Đã xảy ra lỗi trong quá trình hội chẩn: {e}"

        # In kết quả cuối cùng ra màn hình
        st.markdown(final_answer)

    # Lưu câu trả lời của AI vào bộ nhớ
    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# --- 5. NÚT LÀM MỚI LỊCH SỬ CHAT ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử hội chẩn"):
        st.session_state.chat_history = []
        st.rerun()  # Tải lại trang web