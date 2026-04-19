from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Cau hoi y khoa cua nguoi dung")
    role: str = Field(
        default="",
        description="Vai tro nguoi dung tu frontend. role='bac_si_tramyte' se bat shortcut luong tram_y_te.",
    )


class HealthResponse(BaseModel):
    status: str
