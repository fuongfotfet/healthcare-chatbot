import streamlit as st
import asyncio
import uuid
from main import ChatbotApp

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Trợ lý Y khoa Đa đặc vụ", page_icon="🏥", layout="centered")
st.title("🏥 Trợ lý Y khoa Đa đặc vụ")

@st.cache_resource
def load_ai_engine():
    return ChatbotApp()

chatbot_engine = load_ai_engine()

# --- 2. KHỞI TẠO BỘ NHỚ ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Xin chào! Hãy mô tả triệu chứng, tôi sẽ giúp bạn định hướng chuyên khoa nhé."}
    ]

if "pending_routing" not in st.session_state:
    st.session_state.pending_routing = False

config_graph = {"configurable": {"thread_id": st.session_state.thread_id}}

# Hiển thị lại toàn bộ lịch sử
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. XỬ LÝ GIAO DIỆN CHỜ XÁC NHẬN (HUMAN-IN-THE-LOOP) ---
if st.session_state.pending_routing:
    state = chatbot_engine.app.get_state(config_graph)
    current_specialties = state.values.get("analyzed_specialties", [])
    suggested_names = [s["name"] for s in current_specialties]
    
    with st.chat_message("assistant"):
        st.info("🤖 **Lễ tân AI đã phân tích ý định.** Bạn có thể giữ nguyên, thêm hoặc bớt chuyên khoa:")
        
        all_domains = [
            "tim_mach", "ho_hap", "tieu_hoa", "than_kinh", "xuong_khop", 
            "da_lieu", "nhi_khoa", "hiv_aids", "huyet_hoc", "phuc_hoi_chuc_nang", 
            "duoc_hoc", "tam_than", "noi_tiet", "ung_buou", "ky_sinh_trung", 
            "nhiem_khuan", "dinh_duong", "y_te_cong_cong", "truyen_nhiem", 
            "san_phu_khoa", "hau_covid", "cap_cuu"
        ]
        
        selected_domains = st.multiselect(
            "Chọn chuyên khoa hội chẩn:", 
            options=all_domains, 
            default=suggested_names
        )
        
        if st.button("🚀 Xác nhận & Trích xuất dữ liệu"):
            new_specialties = [{"name": name, "is_core_issue": True} for name in selected_domains]
            chatbot_engine.app.update_state(config_graph, {"analyzed_specialties": new_specialties})
            st.session_state.pending_routing = False 
            
            with st.spinner("⚙️ Đang trích xuất dữ liệu và hội chẩn..."):
                try:
                    async def resume_graph():
                        async for _ in chatbot_engine.app.astream(None, config_graph):
                            pass
                    asyncio.run(resume_graph())
                    
                    final_state = chatbot_engine.app.get_state(config_graph)
                    
                    # LOGIC MỚI: Lấy thẳng response từ Backend, không cần bóc JSON nữa
                    final_answer = final_state.values.get("response", "❌ Hệ thống không trả về kết quả.")
                            
                except Exception as e:
                    final_answer = f"❌ Đã xảy ra lỗi: {e}"

                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                st.rerun()

# --- 4. NHẬP CÂU HỎI MỚI ---
elif not st.session_state.pending_routing:
    if user_query := st.chat_input("Ví dụ: Tôi hay bị tê mỏi ngón tay và đau khớp gối..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🕵️ Lễ tân đang phân tích ý định..."):
                try:
                    async def run_to_interrupt():
                        async for _ in chatbot_engine.app.astream({"query": user_query}, config_graph):
                            pass
                    asyncio.run(run_to_interrupt())
                    
                    state = chatbot_engine.app.get_state(config_graph)
                    if state.next and "vector_retrieval" in state.next:
                        st.session_state.pending_routing = True
                        st.rerun()
                    else:
                        # LOGIC MỚI: Lấy thẳng response
                        final_answer = state.values.get("response", "❌ Hệ thống không trả về kết quả.")
                        st.markdown(final_answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                                    
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi: {e}")

# --- 5. NÚT XÓA LỊCH SỬ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử hội chẩn"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4()) 
        st.session_state.pending_routing = False
        st.rerun()