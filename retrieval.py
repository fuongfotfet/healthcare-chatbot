import asyncio
from sentence_transformers import SentenceTransformer
import config
from schemas import RouterState
from database import DatabaseManager


class VectorRetrievalNode:
    def __init__(self):
        print("⏳ [Retriever] Loading Embedding Model...")
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.db_manager = DatabaseManager()

    async def process(self, state: RouterState):
        hyde_text = state.get("hypothetical_document", "")
        domains_data = state.get("analyzed_specialties", [])

        print(f"🔍 [Retriever] Đang quét song song trên {len(domains_data)} domain...")
        query_vec = self.embed_model.encode(hyde_text).tolist()

        # Biến dùng chung để lưu mapping
        global_source_mapping = {}

        def fetch_from_db(domain_item):
            domain_name = domain_item["name"]
            chunk_limit = 5 if domain_item["is_core_issue"] else 3

            try:
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.title, d.source, c.content 
                    FROM document_chunks c JOIN medical_documents d ON c.document_id = d.id
                    WHERE d.specialty = %s ORDER BY c.embedding <=> %s::vector LIMIT %s;
                """, (domain_name, query_vec, chunk_limit))

                results = cursor.fetchall()
                cursor.close()
                conn.close()

                if results:
                    formatted_chunks = []
                    for i, row in enumerate(results):
                        # TẠO ID DUY NHẤT CHO MỖI CHUNK (Ví dụ: [TIM_MACH_1])
                        ref_id = f"[{domain_name.upper()}_{i + 1}]"

                        # Lưu metadata vào từ điển
                        global_source_mapping[ref_id] = {
                            "title": row[0],
                            "source": row[1],
                            "specialty": domain_name,
                            "chunk_text": row[2]
                        }
                        # Đưa nội dung cho LLM chỉ kèm ID
                        formatted_chunks.append(f"{ref_id} NỘI DUNG: {row[2]}")

                    return domain_name, "\n\n".join(formatted_chunks)
                return domain_name, "No data retrieved."
            except Exception as e:
                print(f"❌ [DB Error] {domain_name}: {e}")
                return domain_name, "System error."

        tasks = [asyncio.to_thread(fetch_from_db, spec) for spec in domains_data]
        results = await asyncio.gather(*tasks) if tasks else []

        return {
            "specialty_contexts": {name: context for name, context in results},
            "source_mapping": global_source_mapping  # TRẢ VỀ API CONTEXT MAPPING
        }