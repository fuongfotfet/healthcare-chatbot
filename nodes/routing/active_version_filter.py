from core.database import DatabaseManager
from core.schemas import RouterState


class ActiveVersionFilterNode:
    """Lọc các version active từ guideline_versions theo bệnh đã route."""

    def __init__(self):
        print("⏳ [Version Filter] Initializing...")
        self.db_manager = DatabaseManager()

    def process(self, state: RouterState):
        routed_diseases = state.get("routed_diseases", {})
        analyzed_specialties = state.get("analyzed_specialties", [])

        specialty_values = [item["name"] for item in analyzed_specialties if item.get("name")]

        if not routed_diseases and not specialty_values:
            print("⚠️ [Version Filter] Không có dữ liệu để lọc version active.")
            return {"active_version_ids": []}

        candidate_pairs = [
            (chuyen_khoa, ten_benh)
            for chuyen_khoa, disease_names in routed_diseases.items()
            for ten_benh in disease_names
            if chuyen_khoa and ten_benh
        ]

        # Loại duplicate, giữ thứ tự xuất hiện.
        candidate_pairs = list(dict.fromkeys(candidate_pairs))

        conn = None
        cursor = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            if candidate_pairs:
                pair_conditions = " OR ".join(
                    ["(g.chuyen_khoa = %s AND g.ten_benh = %s)"] * len(candidate_pairs)
                )
                params = [value for pair in candidate_pairs for value in pair]
                cursor.execute(
                    f"""
                    SELECT DISTINCT gv.version_id
                    FROM guideline_versions gv
                    JOIN guidelines g ON g.guideline_id = gv.guideline_id
                    WHERE gv.status = 'active'
                      AND ({pair_conditions})
                    ORDER BY gv.version_id;
                    """,
                    tuple(params),
                )
            else:
                cursor.execute(
                    """
                                        SELECT DISTINCT gv.version_id
                    FROM guideline_versions gv
                    JOIN guidelines g ON g.guideline_id = gv.guideline_id
                    WHERE gv.status = 'active'
                      AND g.chuyen_khoa = ANY(%s)
                    ORDER BY gv.version_id;
                    """,
                    (specialty_values,),
                )

            rows = cursor.fetchall()
            version_ids = [row[0] for row in rows]
            print(f"🧾 [Version Filter] Active version IDs: {version_ids}")
            return {"active_version_ids": version_ids}
        except Exception as e:
            print(f"❌ [Version Filter Error] {e}")
            return {"active_version_ids": []}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
