import os
import sys
import time
import json
import re
import pymongo
from pymongo.errors import PyMongoError
from bson import json_util
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OrionSmartCity")

# --- Configuration (override via environment variables) ---
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", "27037")
DB_NAME = os.getenv("MONGO_DB", "orion")
COLLECTION = os.getenv("MONGO_COLLECTION", "entities")
SEED_FILE = os.getenv("SEED_FILE", "orion.entities.json")


def _log(message: str) -> None:
    # IMPORTANT: with stdio transport, stdout is reserved for the MCP (JSON-RPC)
    # protocol. Any diagnostic output MUST go to stderr, otherwise the client
    # parses our log lines as protocol messages and the connection breaks.
    print(f"[OrionSmartCity] {message}", file=sys.stderr, flush=True)


def _connect(timeout_ms: int = 3000) -> pymongo.MongoClient:
    print("Connection to database...")
    return pymongo.MongoClient(
        f"mongodb://{MONGO_HOST}:{MONGO_PORT}/",
        serverSelectionTimeoutMS=timeout_ms,
    )

def _val(prop):
    """Estrae il valore da una Property NGSI-LD, sciogliendo i DateTime {'@value': ...}."""
    v = prop.get("value") if isinstance(prop, dict) else prop
    return v["@value"] if isinstance(v, dict) and "@value" in v else v


@mcp.tool()
def get_orion_entities(entity_type: str = "*", city: str = None, limit: int = 5) -> str:
    """Read entities from the Orion/MongoDB DB. Returns compact JSON.
    If found is false, the JSON includes suggestions (available_types, pilot_cities)."""
    valid = [c.strip().lower() for c in os.getenv("PILOT_CITIES", "").split(",") if c.strip()]
    client = None
    try:
        client = _connect()
        coll = client[DB_NAME][COLLECTION]

        query = {}
        # _id.type e' l'URI NGSI-LD espanso (.../default-context/DatasetDCAT-AP):
        # match sul termine locale ancorato in coda, case-insensitive.
        if entity_type and entity_type != "*":
            query["_id.type"] = {"$regex": re.compile(re.escape(entity_type) + r"$", re.IGNORECASE)}
        if city:
            if valid and city.lower() not in valid:
                return json.dumps({"found": False, "error": f"'{city}' is not a pilot city",
                                   "pilot_cities": valid})
            query["_id.id"] = {"$regex": re.compile(re.escape(city.lower()), re.IGNORECASE)}

        results = list(coll.find(query).limit(limit))

        # Niente risultati -> restituiamo suggerimenti, non il vuoto.
        if not results:
            types = sorted({t.rsplit("/", 1)[-1] for t in coll.distinct("_id.type")})
            return json.dumps({"found": False, "available_types": types, "pilot_cities": valid})

        # Output compatto: id, type breve e attributi {nome_breve: valore}.
        entities = [{
            "id": d["_id"]["id"],
            "type": d["_id"]["type"].rsplit("/", 1)[-1],
            "attributes": {k.rsplit("/", 1)[-1].rsplit("#", 1)[-1]: _val(v)
                           for k, v in d.get("attrs", {}).items()},
        } for d in results]
        return json.dumps({"found": True, "entities": entities}, default=json_util.default)

    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    mcp.run(transport='stdio')