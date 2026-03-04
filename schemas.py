from typing import TypedDict, List, Dict
from pydantic import BaseModel, Field

class RouterState(TypedDict):
    query: str
    analyzed_specialties: list
    hypothetical_document: str
    specialty_contexts: Dict[str, str]
    specialty_reports: Dict[str, str]
    response: str

class SpecialtyDetail(BaseModel):
    # Đã cập nhật đầy đủ 22 chuyên khoa để LLM biết đường chọn
    name: str = Field(description="Tên chuyên khoa: 'tim_mach', 'ho_hap', 'tieu_hoa', 'than_kinh', 'xuong_khop', 'da_lieu', 'nhi_khoa', 'hiv_aids', 'huyet_hoc', 'phuc_hoi_chuc_nang', 'duoc_hoc', 'tam_than', 'noi_tiet', 'ung_buou', 'ky_sinh_trung', 'nhiem_khuan', 'dinh_duong', 'y_te_cong_cong', 'truyen_nhiem', 'san_phu_khoa', 'hau_covid', 'cap_cuu'.")
    is_core_issue: bool = Field(description="Đánh dấu True nếu là vấn đề cấp cứu hoặc cốt lõi.")

class RouteDecision(BaseModel):
    analyzed_specialties: List[SpecialtyDetail] = Field(description="Danh sách các khoa liên quan.")
    hypothetical_document: str = Field(description="Đoạn văn HyDE tóm tắt triệu chứng, câu hỏi của bệnh nhân.")