import os
import sys
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Hardcoded dictionary for common synonyms to their canonical ticker
SYNONYMS = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "ripple": "xrp",
    "binance coin": "bnb",
    "cardano": "ada",
    "dogecoin": "doge",
    "avalanche": "avax",
    "chainlink": "link",
    "polkadot": "dot",
    "polygon": "matic",
    "shiba inu": "shib",
    "litecoin": "ltc",
    "tether": "usdt",
    "usd coin": "usdc",
    "비트코인": "btc",
    "이더리움": "eth",
    "솔라나": "sol"
}

def merge_nodes(driver, alias: str, canonical: str):
    """Merges the alias node into the canonical node."""
    query = """
    MATCH (canonical:Entity {name: $canonical}), (alias:Entity {name: $alias})
    // Use APOC refactor to merge alias into canonical.
    // If APOC is not installed, we would need to manually rewire edges.
    // Let's assume APOC is available. If not, we will fall back to a manual approach.
    CALL apoc.refactor.mergeNodes([canonical, alias], {
        properties: "combine",
        mergeRels: true
    }) YIELD node
    RETURN node
    """
    
    manual_query = """
    MATCH (alias:Entity {name: $alias})
    MERGE (canonical:Entity {name: $canonical})
    ON CREATE SET canonical = alias, canonical.name = $canonical
    
    // Move all outgoing relationships
    WITH alias, canonical
    MATCH (alias)-[r]->(other)
    MERGE (canonical)-[new_r:RELATES_TO {type: type(r)}]->(other)
    ON CREATE SET new_r = r
    DELETE r
    
    // Move all incoming relationships
    WITH alias, canonical
    MATCH (other)-[r]->(alias)
    MERGE (other)-[new_r:RELATES_TO {type: type(r)}]->(canonical)
    ON CREATE SET new_r = r
    DELETE r
    
    // Finally delete the alias node
    WITH alias
    DELETE alias
    """
    
    with driver.session() as session:
        try:
            # Try APOC first
            session.run(query, canonical=canonical, alias=alias)
            logger.info(f"Merged '{alias}' into '{canonical}' using APOC.")
        except Exception as e:
            logger.warning(f"APOC merge failed: {e}. Falling back to manual merge.")
            try:
                session.run(manual_query, canonical=canonical, alias=alias)
                logger.info(f"Merged '{alias}' into '{canonical}' using manual query.")
            except Exception as e_manual:
                logger.error(f"Manual merge failed for '{alias}' -> '{canonical}': {e_manual}")

def run_deduplication():
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        logger.error("Neo4j credentials not found in environment.")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        # Verify connection
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully.")
        
        for alias, canonical in SYNONYMS.items():
            if alias == canonical:
                continue
            logger.info(f"Checking if '{alias}' exists to merge into '{canonical}'...")
            
            # Check if alias exists
            with driver.session() as session:
                result = session.run("MATCH (n:Entity {name: $alias}) RETURN count(n) as count", alias=alias)
                count = result.single()["count"]
                if count > 0:
                    logger.info(f"Found {count} instance(s) of '{alias}'. Merging...")
                    merge_nodes(driver, alias, canonical)
                else:
                    logger.debug(f"No instance of '{alias}' found.")
                    
        logger.info("Deduplication complete.")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    run_deduplication()
