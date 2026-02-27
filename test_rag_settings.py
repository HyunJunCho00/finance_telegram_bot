import sys
import os

# 프로젝트 루트 경로 추가 (config 패키지를 찾기 위함)
sys.path.append(os.getcwd())

from config.settings import get_settings
from neo4j import GraphDatabase
from pymilvus import MilvusClient
import voyageai

def test_neo4j(settings):
    print("\n--- 1. Testing Neo4j Connection ---")
    uri = settings.neo4j_uri
    password = settings.NEO4J_PASSWORD
    user = "neo4j"
    
    # Debug info (masked)
    if uri:
        print(f"Target URI: {uri}")
    if password:
        masked_pwd = password[0] + "*" * (len(password)-2) + password[-1] if len(password) > 2 else "***"
        print(f"Password Loaded (Masked): {masked_pwd} (Length: {len(password)})")

    # Attempt 1: Standard
    try:
        print(f"Attempting standard connection: {uri}")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("SUCCESS: Neo4j Connected successfully!")
        driver.close()
        return True
    except Exception as e:
        print(f"FAILED: Standard Connection Failed: {e}")
        
    # Attempt 2: Diagnostic SSL Bypass
    print("\nDIAGNOSTIC: Attempting WITHOUT SSL verification...")
    try:
        insecure_uri = uri.replace("neo4j+s", "neo4j").replace("bolt+s", "bolt")
        driver = GraphDatabase.driver(insecure_uri, auth=(user, password), trust="TRUST_ALL_CERTIFICATES")
        driver.verify_connectivity()
        print("SUCCESS: Connected successfully (SSL Bypassed)!")
        driver.close()
        return True
    except Exception as e:
        print(f"FAILED: SSL Bypass also failed: {e}")
    return False

def test_milvus(settings):
    print("\n--- 2. Testing Milvus (Zilliz) Connection ---")
    uri = settings.MILVUS_URI
    token = settings.MILVUS_TOKEN
    
    if not uri or not token:
        print("[FAIL] Skip: MILVUS_URI or MILVUS_TOKEN missing in Settings")
        return False

    try:
        client = MilvusClient(uri=uri, token=token)
        collections = client.list_collections()
        print("SUCCESS: Milvus Connected successfully!")
        print(f"   Existing collections: {collections}")
        return True
    except Exception as e:
        print(f"FAILED: Milvus Connection Failed: {e}")
        return False

if __name__ == "__main__":
    # VM에서 실행 시 USE_SECRET_MANAGER=true 환경변수가 필요할 수 있음
    print(f"Current USE_SECRET_MANAGER: {os.getenv('USE_SECRET_MANAGER', 'false')}")
    
    try:
        settings = get_settings()
        print("Settings loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load settings: {e}")
        sys.exit(1)

    n_ok = test_neo4j(settings)
    m_ok = test_milvus(settings)
    
    print("\n" + "="*30)
    if n_ok and m_ok:
        print("SYSTEMS GO: RAG is ready to use on VM!")
    else:
        print("SYSTEMS FAILED: Check your Secret Manager or firewall.")
    print("="*30)
