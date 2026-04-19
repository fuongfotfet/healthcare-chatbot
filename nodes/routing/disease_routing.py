import asyncio
from langchain_openai import ChatOpenAI
from core import config
from core.schemas import RouterState, SpecialtyDiseaseDecision
from core.prompts import DISEASE_ROUTING_PROMPT
from core.database import DatabaseManager


class DiseaseRoutingNode:
    """Node định tuyến bệnh theo ten_benh từ database."""

    def __init__(self):
        print("⏳ [Disease Router] Initializing...")
        self.llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)
        self.db_manager = DatabaseManager()

    def _load_disease_candidates(self, specialty_name: str):
        conn = None
        cursor = None
        try:
            if not specialty_name:
                return []

            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT ten_benh
                FROM guidelines
                WHERE chuyen_khoa = %s
                  AND ten_benh IS NOT NULL
                  AND btrim(ten_benh) <> ''
                ORDER BY ten_benh;
                """,
                (specialty_name,),
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"❌ [Disease Router DB Error] {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    async def process(self, state: RouterState):
        query = state.get("query", "")
        analyzed_specialties = state.get("analyzed_specialties", [])

        specialty_names = [item.get("name") for item in analyzed_specialties if item.get("name")]
        specialty_names = list(dict.fromkeys(specialty_names))

        if not specialty_names:
            return {
                "routed_diseases": {},
            }

        async def route_single_specialty(specialty_name: str):
            candidates = self._load_disease_candidates(specialty_name)
            if not candidates:
                print(f"⚠️ [Disease Router] Không có ứng viên bệnh cho khoa {specialty_name}.")
                return specialty_name, []

            candidates_string = "\n".join([f"- {specialty_name}: {disease}" for disease in candidates])
            prompt = DISEASE_ROUTING_PROMPT.format(
                specialties_string=specialty_name,
                candidates_string=candidates_string,
                query=query,
            )

            structured_llm = self.llm.with_structured_output(SpecialtyDiseaseDecision)
            try:
                decision = await structured_llm.ainvoke(prompt)
                valid_diseases = set(candidates)
                selected = []
                seen = set()

                for disease_name in decision.ten_benh:
                    if disease_name in valid_diseases and disease_name not in seen:
                        seen.add(disease_name)
                        selected.append(disease_name)

                if not selected:
                    print(f"⚠️ [Disease Router] {specialty_name}: LLM không trả bệnh hợp lệ, fallback về candidates DB.")
                    return specialty_name, candidates

                return specialty_name, selected
            except Exception as e:
                print(f"❌ [Disease Router Error] {specialty_name}: {e}")
                return specialty_name, candidates

        tasks = [route_single_specialty(name) for name in specialty_names]
        results = await asyncio.gather(*tasks) if tasks else []

        grouped_diseases = {specialty: diseases for specialty, diseases in results if diseases}

        if not grouped_diseases:
            print("⚠️ [Disease Router] Không route được bệnh hợp lệ từ tất cả chuyên khoa.")
            return {"routed_diseases": {}}

        total_diseases = sum(len(diseases) for diseases in grouped_diseases.values())
        print(f"🎯 [Disease Router] Tổng số bệnh route được: {total_diseases}")
        return {
            "routed_diseases": grouped_diseases,
        }
