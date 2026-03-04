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
    name: str = Field(description="Tên chuyên khoa: 'tim_mach', 'ho_hap', 'tieu_hoa', 'than_kinh', 'xuong_khop'.")
    is_core_issue: bool = Field(description="Đánh dấu True nếu là vấn đề cấp cứu hoặc cốt lõi.")

class RouteDecision(BaseModel):
    analyzed_specialties: List[SpecialtyDetail] = Field(description="Danh sách các khoa liên quan.")
    hypothetical_document: str = Field(description="Đoạn văn HyDE tóm tắt triệu chứng, câu hỏi của bệnh nhân.")