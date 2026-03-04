import asyncio
from sentence_transformers import SentenceTransformer
import config
from schemas import RouterState
from database import DatabaseManager


class VectorRetrievalNode:
    """Class đảm nhiệm truy xuất dữ liệu từ Vector Database (RAG)"""

    def __init__(self):
        print("⏳ [Retriever] Loading Embedding Model...")
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.db_manager = DatabaseManager()

    async def process(self, state: RouterState):
        hyde_text = state["hypothetical_document"]
        domains_data = state["analyzed_specialties"]

        print(f"🔍 [Retriever] Đang quét song song trên {len(domains_data)} domain...")
        query_vec = self.embed_model.encode(hyde_text).tolist()

        def fetch_from_db(domain_item):
            domain_name = domain_item["name"]
            chunk_limit = 5 if domain_item["is_core_issue"] else 2

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
                    chunks = [f"📌 Source: {row[0]}\n📝 Content: {row[2]}" for row in results]
                    return domain_name, "\n\n".join(chunks)
                return domain_name, "No data retrieved."
            except Exception as e:
                print(f"❌ [DB Error] {domain_name}: {e}")
                return domain_name, "System error."

        tasks = [asyncio.to_thread(fetch_from_db, spec) for spec in domains_data]
        results = await asyncio.gather(*tasks) if tasks else []

        return {"specialty_contexts": {name: context for name, context in results}}