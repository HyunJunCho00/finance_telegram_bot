import os
import json
from neo4j import GraphDatabase
from pymilvus import MilvusClient
import voyageai
from dotenv import load_dotenv

# .env 로드
load_dotenv()

def test_neo4j():
    print("\n--- 1. Testing Neo4j Connection ---")
    uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
    password = os.getenv("NEO4J_PASSWORD")
    user = "neo4j"
    
    if not uri or not password:
        print("[FAIL] Skip: NEO4J_URI or NEO4J_PASSWORD missing in .env")
        return False

    # Attempt 1: Standard Connection
    try:
        print("Attempting standard connection (SSL enabled)...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("SUCCESS: Neo4j Connected successfully!")
        driver.close()
        return True
    except Exception as e:
        print(f"FAILED: Standard Connection Failed: {e}")
        
    # Attempt 2: Disable SSL Verification (DIAGNOSTIC ONLY)
    print("\nDIAGNOSTIC: Attempting connection WITHOUT SSL verification...")
    try:
        # For SSL bypass, we must use prefix 'neo4j://' (not +s)
        insecure_uri = uri.replace("neo4j+s", "neo4j").replace("bolt+s", "bolt")
        driver = GraphDatabase.driver(insecure_uri, auth=(user, password), trust="TRUST_ALL_CERTIFICATES")
        driver.verify_connectivity()
        print("SUCCESS: Connected successfully (SSL Verification Bypassed)!")
        print("TIP: This means the issue is strictly with your local SSL certificate store.")
        driver.close()
        return True
    except Exception as e:
        print(f"FAILED: Connection without SSL verification also failed: {e}")

    print("\nTIP: Try running 'pip install certifi' or check if your firewall blocks port 7687.")
    return False

def test_milvus():
    print("\n--- 2. Testing Milvus (Zilliz) Connection ---")
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    
    if not uri or not token:
        print("[FAIL] Skip: MILVUS_URI or MILVUS_TOKEN missing in .env")
        return False

    try:
        client = MilvusClient(uri=uri, token=token)
        # Check connection by listing collections
        collections = client.list_collections()
        print("SUCCESS: Milvus Connected successfully!")
        print(f"   Existing collections: {collections}")
        return True
    except Exception as e:
        print(f"FAILED: Milvus Connection Failed: {e}")
        print("   (Tip: Check if your Zilliz cluster is 'Suspended')")
        return False

def test_voyage_ai():
    print("\n--- 3. Testing Voyage AI Embedding (Optional) ---")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("[FAIL] Skip: VOYAGE_API_KEY missing in .env")
        return False

    try:
        vo = voyageai.Client(api_key=api_key)
        result = vo.embed(["Test connectivity"], model="voyage-finance-2")
        print(f"SUCCESS: Voyage AI Embedding working! (Dim: {len(result.embeddings[0])})")
        return True
    except Exception as e:
        print(f"FAILED: Voyage AI Failed: {e}")
        return False

if __name__ == "__main__":
    n_ok = test_neo4j()
    m_ok = test_milvus()
    v_ok = test_voyage_ai()
    
    print("\n" + "="*30)
    if n_ok and m_ok:
        print("SYSTEMS GO: RAG is ready to use!")
    else:
        print("SYSTEMS FAILED: Check your credentials and instance status.")
    print("="*30)
