# --- code/local_export_all.py ---
"""
use: python code/local_export_all.py
# This script requires a Neo4j database with the full KG.
# Reviewers can skip this file and use the pre-exported data in data/ directly.
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import json
import time
import sys
import random

# ================= set =================
NEO4J_URI = "URI"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 50000


class FullDataExporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
    def close(self):
        self.driver.close()

    def export_5a_training_data(self):
        """(Gold + Hard Negatives)
        Required fields (5a run_5a() dependency):
        - doc_text, pat_text: Used for Cross-Encoder computation
        - label: Gold standard is 1, negative sampling is 0
        - score_company, score_vector, score_entity, total_score: Feature engineering
        - reasons: Extracting features such as TierS/Rescue/sim_raw
        - doc_company, pat_company: Company matching features
        - fda_id, pat_id: Used for debugging and tracing
        """
        print("\n" + "="*60)
        print("[1/7] Exporting 5a Training Data (Gold + Hard Negatives)...")
        print("="*60)
        start = time.time()
        
        # ==================== Gold Standard ====================
        query_gold = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        WHERE (d)-[:APPLICANT_IS]->() AND (p)-[:ASSIGNED_TO]->()
        
        OPTIONAL MATCH (d)-[r:V4_WEIGHTED_LINK]-(p)
        OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c1:COMPANY)
        OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c2:COMPANY)
        
        RETURN DISTINCT
            d.PMANUMBER as fda_id,
            p.patent_id as pat_id,
            coalesce(d.DEVICENAME, '') + ' ' + coalesce(d.text_snippet, '') as doc_text,
            coalesce(p.patent_title, '') + ' ' + coalesce(p.abstract, '') as pat_text,
            r.score as total_score,
            r.score_company as score_company,
            r.score_vector as score_vector,
            r.score_entity as score_entity,
            r.reasons as reasons,
            c1.name as doc_company,
            c2.name as pat_company,
            1 as label
        """
        
        with self.driver.session() as session:
            gold_data = session.run(query_gold).data()
        
        n_gold = len(gold_data)
        print(f"   Gold Standard (Positive samples): {n_gold:,}")
        
        # ====================  Hard Negatives ====================
        n_neg_target = n_gold * 3
        
        print(f"Sampling negative samples (target: {n_neg_target:,})...")
        print(f"(Optimization: Batch sampling avoids full table scan)")
        
        gold_pairs = set()
        for row in gold_data:
            gold_pairs.add((row['fda_id'], row['pat_id']))
        
        query_neg = """
        MATCH (d:DOCUMENT)-[r:V4_WEIGHTED_LINK]-(p:PATENT)
        WHERE r.score >= 55
          AND r.score < $score_upper
        
        OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c1:COMPANY)
        OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c2:COMPANY)
        
        WITH d, p, r, c1, c2
        WHERE c1 IS NOT NULL AND c2 IS NOT NULL
        
        RETURN 
            d.PMANUMBER as fda_id,
            p.patent_id as pat_id,
            coalesce(d.DEVICENAME, '') + ' ' + coalesce(d.text_snippet, '') as doc_text,
            coalesce(p.patent_title, '') + ' ' + coalesce(p.abstract, '') as pat_text,
            r.score as total_score,
            r.score_company as score_company,
            r.score_vector as score_vector,
            r.score_entity as score_entity,
            r.reasons as reasons,
            c1.name as doc_company,
            c2.name as pat_company,
            0 as label
        LIMIT 10000
        """
        
        neg_data = []
        #  55-65, 65-75, 75-85, 85-95, 95+
        score_ranges = [(55, 65), (65, 75), (75, 85), (85, 95), (95, 200)]
        samples_per_range = n_neg_target // len(score_ranges) + 1
        
        with self.driver.session() as session:
            for score_lower, score_upper in score_ranges:
                if len(neg_data) >= n_neg_target:
                    break
                    
                range_query = f"""
                MATCH (d:DOCUMENT)-[r:V4_WEIGHTED_LINK]-(p:PATENT)
                WHERE r.score >= {score_lower} AND r.score < {score_upper}
                
                OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c1:COMPANY)
                OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c2:COMPANY)
                
                WITH d, p, r, c1, c2
                WHERE c1 IS NOT NULL AND c2 IS NOT NULL
                
                RETURN 
                    d.PMANUMBER as fda_id,
                    p.patent_id as pat_id,
                    coalesce(d.DEVICENAME, '') + ' ' + coalesce(d.text_snippet, '') as doc_text,
                    coalesce(p.patent_title, '') + ' ' + coalesce(p.abstract, '') as pat_text,
                    r.score as total_score,
                    r.score_company as score_company,
                    r.score_vector as score_vector,
                    r.score_entity as score_entity,
                    r.reasons as reasons,
                    c1.name as doc_company,
                    c2.name as pat_company,
                    0 as label
                LIMIT {samples_per_range * 2}
                """
                
                batch_data = session.run(range_query).data()
                
                # filter Gold pairs
                filtered = [
                    row for row in batch_data 
                    if (row['fda_id'], row['pat_id']) not in gold_pairs
                ]
                
                # Random sampling
                random.shuffle(filtered)
                neg_data.extend(filtered[:samples_per_range])
                
                print(f"  Score {score_lower}-{score_upper}: obtained {len(filtered)} , sampling {min(len(filtered), samples_per_range)} ")
        
        # Final cutoff to target quantity
        neg_data = neg_data[:n_neg_target]
        n_neg = len(neg_data)
        print(f"   Hard Negatives: {n_neg:,}")
        
        all_data = gold_data + neg_data
        df = pd.DataFrame(all_data)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for col in ['total_score', 'score_company', 'score_vector', 'score_entity']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        for col in ['reasons', 'doc_company', 'pat_company', 'doc_text', 'pat_text']:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        output_file = OUTPUT_DIR / "training_data_5a.parquet"
        df.to_parquet(output_file, index=False)
        
        pos_count = (df['label'] == 1).sum()
        neg_count = (df['label'] == 0).sum()
        
        print(f"\n    5a Training Data: {len(df):,} rows")
        print(f"      Positive (Gold): {pos_count:,}")
        print(f"      Negative (Hard): {neg_count:,}")
        print(f"      Ratio: 1:{neg_count/pos_count:.1f}")
        print(f"      Time: {time.time()-start:.1f}s")

        required_cols = ['fda_id', 'pat_id', 'doc_text', 'pat_text', 
                        'total_score', 'score_company', 'score_vector', 'score_entity',
                        'reasons', 'doc_company', 'pat_company', 'label']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"   ⚠️ Warning: Missing columns: {missing}")
        else:
            print(f"   ✅ All required columns present")

    def export_5b_links(self):
        print("\n" + "="*60)
        print("[2/7] Exporting 5b links (for AI Reranking)...")
        print("      This may take 10-30 minutes for large datasets...")
        start = time.time()

        count_query = """
        MATCH (d:DOCUMENT)<-[r:V4_WEIGHTED_LINK]-(p:PATENT)
        WHERE r.checked_by_ai_v2 IS NULL
          AND (r.status = 'ACCEPTED' OR r.score_company > 0)
        RETURN count(r) as total
        """
        
        with self.driver.session() as session:
            total = session.run(count_query).single()['total']
            print(f"      Total links to export: {total:,}")
        
        query = """
        MATCH (d:DOCUMENT)<-[r:V4_WEIGHTED_LINK]-(p:PATENT)
        WHERE r.checked_by_ai_v2 IS NULL
          AND (r.status = 'ACCEPTED' OR r.score_company > 0)
        
        OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c1:COMPANY)
        OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c2:COMPANY)
        
        RETURN 
            elementId(r) as rel_id,
            coalesce(d.DEVICENAME, '') + ' ' + coalesce(d.text_snippet, '') as doc_text,
            coalesce(p.patent_title, '') + ' ' + coalesce(p.abstract, '') as pat_text,
            r.score as total_score,
            r.score_company as score_company,
            r.score_vector as score_vector,
            r.score_entity as score_entity,
            r.reasons as reasons,
            c1.name as doc_company,
            c2.name as pat_company
        """
        
        output_file = OUTPUT_DIR / "links_to_process.parquet"
        parquet_writer = None
        total_count = 0
        last_print = time.time()
        
        with self.driver.session() as session:
            result = session.run(query)
            batch = []
            
            for record in result:
                batch.append(record.data())
                
                if len(batch) >= BATCH_SIZE:
                    df_chunk = pd.DataFrame(batch)
                    table = pa.Table.from_pandas(df_chunk)
                    
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(str(output_file), table.schema)
                    
                    parquet_writer.write_table(table)
                    total_count += len(batch)
                    
                    if time.time() - last_print > 10:
                        elapsed = time.time() - start
                        speed = total_count / elapsed
                        eta = (total - total_count) / speed if speed > 0 else 0
                        print(f"      Progress: {total_count:,}/{total:,} ({total_count/total*100:.1f}%) - ETA: {eta/60:.1f} min")
                        last_print = time.time()
                        sys.stdout.flush()
                    
                    batch = []
            
            if batch:
                df_chunk = pd.DataFrame(batch)
                table = pa.Table.from_pandas(df_chunk)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(str(output_file), table.schema)
                parquet_writer.write_table(table)
                total_count += len(batch)
                
        if parquet_writer:
            parquet_writer.close()
        
        print(f"   5b links: {total_count:,} rows ({time.time()-start:.1f}s)")

    def export_gold_rel_ids(self):
        print("\n" + "="*60)
        print("[3/7] Exporting Gold rel_ids (for 5b monitoring)...")
        start = time.time()
        
        query = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        MATCH (d)<-[r:V4_WEIGHTED_LINK]-(p)
        RETURN DISTINCT elementId(r) as rel_id
        """
        
        with self.driver.session() as session:
            data = session.run(query).data()
        
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_DIR / "gold_rel_ids.csv", index=False)
        
        print(f"   Gold rel_ids: {len(df):,} rows ({time.time()-start:.1f}s)")
        
        count_query = """
        MATCH (d:DOCUMENT)-[r:GOLD_STANDARD_LINK]-(p:PATENT)
        RETURN count(r) as total
        """
        with self.driver.session() as session:
            total_gold = session.run(count_query).single()['total']
            print(f"      Note: Total Gold Standard links: {int(total_gold)}")
            print(f"      Matched in V4_WEIGHTED_LINK: {len(df)}")

    def export_gold_standard(self):
        print("\n" + "="*60)
        print("[4/7] Exporting Gold Standard (with vectors)...")
        start = time.time()
        
        query = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        
        OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c1:COMPANY)
        OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c2:COMPANY)
        
        OPTIONAL MATCH (d)-[:MENTIONS]->(e1) 
        WHERE e1:LINK_COMPONENT OR e1:LINK_MECHANISM
        OPTIONAL MATCH (p)-[:MENTIONS]->(e2) 
        WHERE e2:LINK_COMPONENT OR e2:LINK_MECHANISM
        
        OPTIONAL MATCH (d)-[kg:V4_WEIGHTED_LINK]-(p)
        
        WITH d, p, c1, c2, kg,
             collect(DISTINCT e1.name) as ent_names_d,
             collect(DISTINCT e2.name) as ent_names_p,
             p.abstract IS NOT NULL AND d.text_snippet IS NOT NULL AND
             EXISTS { MATCH (p)-[:MENTIONS]->() } AND EXISTS { MATCH (d)-[:MENTIONS]->() } AND
             EXISTS { MATCH (p)-[:ASSIGNED_TO]->() } AND EXISTS { MATCH (d)-[:APPLICANT_IS]->() } as is_valid
        
        RETURN 
            d.PMANUMBER as fda_id, 
            p.patent_id as pat_id,
            d.embedding as vec_d,
            p.embedding as vec_p,
            d.text_snippet as doc_text, 
            p.abstract as pat_text,
            c1.name as comp_d, 
            c2.name as comp_p,
            ent_names_d, 
            ent_names_p,
            is_valid,
            kg IS NOT NULL as kg_linked, 
            kg.score as kg_score, 
            kg.reasons as kg_reasons
        """
        
        with self.driver.session() as session:
            data = session.run(query).data()
        
        df = pd.DataFrame(data)
        
        def vector_to_string(v):
            """将向量列表转为逗号分隔的字符串"""
            if v is None:
                return ''
            if isinstance(v, (list, np.ndarray)):
                return ','.join([str(float(x)) for x in v])
            return str(v)
        
        if 'vec_d' in df.columns:
            df['vec_d'] = df['vec_d'].apply(vector_to_string)
        if 'vec_p' in df.columns:
            df['vec_p'] = df['vec_p'].apply(vector_to_string)


        for col in ['ent_names_d', 'ent_names_p']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: '|||'.join([str(i) for i in x]) if x else '')
        
        df.to_parquet(OUTPUT_DIR / "gold_standard.parquet", index=False)
        

        vec_d_count = (df['vec_d'] != '').sum() if 'vec_d' in df.columns else 0
        vec_p_count = (df['vec_p'] != '').sum() if 'vec_p' in df.columns else 0
        
        print(f"   Gold Standard: {len(df):,} rows, valid: {df['is_valid'].sum():,}")
        print(f"     Vector coverage: doc={vec_d_count}/{len(df)}, pat={vec_p_count}/{len(df)}")
        print(f"     Time: {time.time()-start:.1f}s")

    def export_evaluation_dataset(self):
        print("\n" + "="*60)
        print("[5/7] Exporting Evaluation Dataset (for Step 6)...")
        start = time.time()
        
        # Gold Standard
        query_pos = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        WHERE d.text_snippet IS NOT NULL AND p.abstract IS NOT NULL
        RETURN DISTINCT
            d.PMANUMBER as fda_id,
            p.patent_id as pat_id,
            coalesce(d.DEVICENAME, '') + '. ' + d.text_snippet as doc_full_text,
            coalesce(p.patent_title, '') + '. ' + p.abstract as pat_full_text,
            1 as label
        """
        
        # 随机采样非 Gold 的高分链接
        query_neg = """
        MATCH (d:DOCUMENT)<-[r:V4_WEIGHTED_LINK]-(p:PATENT)
        WHERE NOT EXISTS { MATCH (d)-[:GOLD_STANDARD_LINK]-(p) }
          AND d.text_snippet IS NOT NULL 
          AND p.abstract IS NOT NULL
          AND r.score >= 70
        WITH d, p, r, rand() as rnd
        ORDER BY rnd
        LIMIT 2000
        RETURN 
            d.PMANUMBER as fda_id,
            p.patent_id as pat_id,
            coalesce(d.DEVICENAME, '') + '. ' + d.text_snippet as doc_full_text,
            coalesce(p.patent_title, '') + '. ' + p.abstract as pat_full_text,
            0 as label
        """
        
        with self.driver.session() as session:
            pos_data = session.run(query_pos).data()
            neg_data = session.run(query_neg).data()
        
        df_pos = pd.DataFrame(pos_data)
        df_neg = pd.DataFrame(neg_data)
        df = pd.concat([df_pos, df_neg], ignore_index=True)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        df.to_csv(OUTPUT_DIR / "evaluation_dataset.csv", index=False)
        
        print(f"    Evaluation Dataset: {len(df):,} rows (pos={len(df_pos)}, neg={len(df_neg)})")
        print(f"    Time: {time.time()-start:.1f}s")

    def export_baseline_data(self):
        print("\n" + "="*60)
        print("[6/7] Exporting Baseline data...")
        start = time.time()
        
        # ==================== Step 1: ID ====================
        q_gold_patents = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        WHERE p.abstract IS NOT NULL AND p.patent_id IS NOT NULL
        RETURN DISTINCT p.patent_id as id
        """
        
        with self.driver.session() as session:
            gold_patent_ids = [r['id'] for r in session.run(q_gold_patents).data()]
        
        print(f"      Gold patents number: {len(gold_patent_ids)}")
        
        # ==================== Step 2: Gold ====================
        q_gold_patents_full = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        WHERE p.abstract IS NOT NULL AND p.patent_id IS NOT NULL
        OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c:COMPANY)
        RETURN DISTINCT 
            p.patent_id as id, 
            coalesce(p.patent_title, '') + ' ' + p.abstract as text,
            c.name as company
        """
        
        with self.driver.session() as session:
            gold_patents_data = session.run(q_gold_patents_full).data()
        
        gold_patents_df = pd.DataFrame(gold_patents_data)
        print(f"  Gold patents export: {len(gold_patents_df)}")
        
        # ==================== Step 3====================
        remaining = 50000 - len(gold_patents_df)
        
        if remaining > 0:
            gold_ids_str = ', '.join([f"'{pid}'" for pid in gold_patent_ids])
            
            q_other_patents = f"""
            MATCH (p:PATENT) 
            WHERE p.abstract IS NOT NULL 
              AND p.patent_id IS NOT NULL
              AND NOT p.patent_id IN [{gold_ids_str}]
            WITH p, rand() as r 
            ORDER BY r 
            LIMIT {remaining}
            OPTIONAL MATCH (p)-[:ASSIGNED_TO]->(c:COMPANY)
            RETURN p.patent_id as id, 
                   coalesce(p.patent_title, '') + ' ' + p.abstract as text,
                   c.name as company
            """
            
            with self.driver.session() as session:
                other_patents_data = session.run(q_other_patents).data()
            
            other_patents_df = pd.DataFrame(other_patents_data)
            print(f"other patents: {len(other_patents_df)}")
            
            patents = pd.concat([gold_patents_df, other_patents_df], ignore_index=True)
        else:
            patents = gold_patents_df

        patents = patents.drop_duplicates(subset=['id'])
        print(f" Final number of patents: {len(patents)}")
        
        # ==================== Step 4: FDA Docs ====================
        q_fda = """
        MATCH (d:DOCUMENT) 
        WHERE d.text_snippet IS NOT NULL AND d.PMANUMBER IS NOT NULL
        OPTIONAL MATCH (d)-[:APPLICANT_IS]->(c:COMPANY)
        RETURN d.PMANUMBER as id, 
               coalesce(d.DEVICENAME, '') + ' ' + d.text_snippet as text,
               c.name as company
        """
        
        # ==================== Step 5: Gold Links ====================
        q_gold = """
        MATCH (d:DOCUMENT)-[:GOLD_STANDARD_LINK]-(p:PATENT)
        RETURN DISTINCT d.PMANUMBER as fda_id, p.patent_id as pat_id
        """
        
        with self.driver.session() as session:
            fda_docs = pd.DataFrame(session.run(q_fda).data())
            gold = pd.DataFrame(session.run(q_gold).data())
        
        # ==================== Step 6====================
        patent_ids_set = set(patents['id'])
        gold_pat_ids_set = set(gold['pat_id'])
        overlap = gold_pat_ids_set & patent_ids_set
        coverage = len(overlap) / len(gold_pat_ids_set) * 100 if len(gold_pat_ids_set) > 0 else 0
        
        print(f"  Gold patents : {len(overlap)}/{len(gold_pat_ids_set)} ({coverage:.1f}%)")
        
        if coverage < 100:
            print(f"      Warning: some Gold patents did not include!")
            missing = gold_pat_ids_set - patent_ids_set
            print(f"      Missing: {list(missing)[:5]}...")
        
        patents.to_parquet(OUTPUT_DIR / "baseline_patents.parquet", index=False)
        fda_docs.to_parquet(OUTPUT_DIR / "baseline_fda_docs.parquet", index=False)
        gold.to_parquet(OUTPUT_DIR / "baseline_gold_links.parquet", index=False)
        
        print(f"   Patents: {len(patents):,}, FDA: {len(fda_docs):,}, Gold: {len(gold):,} ({time.time()-start:.1f}s)")
        print(f"   Gold patent coverage: {coverage:.1f}%")

    def export_extra_data(self):
        print("\n" + "="*60)
        print("[7/7] Exporting extra data and statistics...")
        start = time.time()
        
        stats_queries = {
            'total_patents': "MATCH (p:PATENT) RETURN count(p) as cnt",
            'total_documents': "MATCH (d:DOCUMENT) RETURN count(d) as cnt",
            'total_gold_links': "MATCH (:DOCUMENT)-[r:GOLD_STANDARD_LINK]->(:PATENT) RETURN count(r) as cnt",
            'total_kg_links': "MATCH (:PATENT)-[r:V4_WEIGHTED_LINK]->(:DOCUMENT) RETURN count(r) as cnt",
            'total_companies': "MATCH (c:COMPANY) RETURN count(c) as cnt",
        }
        
        with self.driver.session() as session:
            stats = {}
            for name, q in stats_queries.items():
                result = session.run(q).single()
                stats[name] = int(result['cnt']) if result else 0
                print(f"      {name}: {stats[name]:,}")
            
            with open(OUTPUT_DIR / "summary_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
        
        print(f"   Stats exported ({time.time()-start:.1f}s)")

    def run(self):
        total_start = time.time()
        
        print("\n" + "="*60)
        print(" EXPORT ALL DATA FOR HPC")
        print("="*60)
        print(f"Output: {OUTPUT_DIR}\n")
        
        self.export_5a_training_data() 
        
        self.export_5b_links()
        self.export_gold_rel_ids()
        self.export_gold_standard()
        self.export_evaluation_dataset()
        self.export_baseline_data()
        self.export_extra_data()
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"ALL DONE! Total time: {total_time/60:.1f} minutes")
        print("="*60)
        
        print("\n Generated files:")
        total_size = 0
        for f in sorted(OUTPUT_DIR.glob("*")):
            size = f.stat().st_size / (1024*1024)
            total_size += size
            print(f"   {f.name:<35} {size:>8.1f} MB")
        print(f"   {'TOTAL':<35} {total_size:>8.1f} MB")
        
        print("\n Verification checklist:")
        required_files = [
            ("training_data_5a.parquet", "5a Reranker training data"),
            ("links_to_process.parquet", "5b AI Reranking input"),
            ("gold_rel_ids.csv", "5b Gold monitoring"),
            ("gold_standard.parquet", "Step 8/10/11 Gold data"),
            ("evaluation_dataset.csv", "Step 6 Reranker comparison"),
            ("baseline_patents.parquet", "Step 7 Baseline"),
            ("baseline_fda_docs.parquet", "Step 7 Baseline"),
            ("baseline_gold_links.parquet", "Step 7 Baseline"),
        ]
        
        all_ok = True
        for fname, desc in required_files:
            path = OUTPUT_DIR / fname
            if path.exists():
                size = path.stat().st_size / (1024*1024)
                print(f"  {fname:<35} {size:>6.1f} MB  ({desc})")
            else:
                print(f"  {fname:<35} MISSING!  ({desc})")
                all_ok = False
        
        if all_ok:
            print("\n All files ready!")
        else:
            print("\n Some files are missing!")

if __name__ == "__main__":
    exporter = FullDataExporter()
    try:
        exporter.run()
    finally:
        exporter.close()