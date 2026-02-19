# --- code/local_import_results.py ---
"""
After downloading the results from HPC, import them back into Neo4j.
"""

import pyarrow.parquet as pq
from neo4j import GraphDatabase
from tqdm import tqdm
import time
from pathlib import Path

NEO4J_URI = "URI"
NEO4J_USER = "USER"
NEO4J_PASSWORD = "PASSWORD"

RESULTS_DIR = Path("./results")
INPUT_FILE = RESULTS_DIR / "final_kg_strict.parquet"

BATCH_SIZE = 5000


def import_results():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    query_update = """
    UNWIND $batch as row
    MATCH ()-[r:V4_WEIGHTED_LINK]-() 
    WHERE elementId(r) = row.rel_id
    SET r.checked_by_ai_v2 = true,
        r.ai_score = row.ai_score,
        r.fusion_prob = row.fusion_prob,
        r.ai_verdict = row.verdict,
        r.ai_updated = datetime()
    """
    
    parquet_file = pq.ParquetFile(str(INPUT_FILE))
    total_rows = parquet_file.metadata.num_rows
    print(f"Total data count: {total_rows:,} rows")
    
    needed_columns = ['rel_id', 'ai_score', 'fusion_prob', 'verdict']
    
    processed_count = 0
    start_time = time.time()
    
    with driver.session() as session:
        for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=needed_columns), 
                          total=(total_rows // BATCH_SIZE) + 1,
                          desc="Importing"):
            
            df_batch = batch.to_pandas()
            batch_data = df_batch.to_dict('records')
            
            if not batch_data:
                continue
            
            session.run(query_update, batch=batch_data)
            processed_count += len(batch_data)
    
    driver.close()
    
    elapsed = time.time() - start_time
    print(f"   used time: {elapsed/60:.1f} minutes")
    print(f"   updated edges: {processed_count:,}") 

def verify_import():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    queries = {
        'Number of links retained by AI': """
            MATCH ()-[r:V4_WEIGHTED_LINK]-() 
            WHERE r.checked_by_ai_v2 = true 
            RETURN count(r)/2 as cnt
        """,
        'Unprocessed links': """
            MATCH ()-[r:V4_WEIGHTED_LINK]-() 
            WHERE r.checked_by_ai_v2 IS NULL 
            RETURN count(r)/2 as cnt
        """,
        'Verdict distribution': """
            MATCH ()-[r:V4_WEIGHTED_LINK]-()
            WHERE r.ai_verdict IS NOT NULL
            RETURN r.ai_verdict as verdict, count(r)/2 as cnt
            ORDER BY cnt DESC
        """,
    }
    
    print("="*60)
    
    with driver.session() as session:
        for name, query in queries.items():
            result = session.run(query)
            print(f"\n{name}:")
            for record in result:
                print(f"   {dict(record)}")
    
    driver.close()


if __name__ == "__main__":
    import_results()
    verify_import()