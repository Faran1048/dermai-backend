import os
import json
from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict

router = APIRouter(prefix="/mock-supabase", tags=["Mock Supabase"])

BASE_DIR = os.path.join(os.getcwd(), "tmp", "mock-supabase")

def get_file_path(table: str, id_val: str) -> str:
    folder = os.path.join(BASE_DIR, table)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{id_val}.json")

@router.get("/{table}/{id_val}")
async def get_record(table: str, id_val: str) -> Dict[str, Any]:
    path = get_file_path(table, id_val)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {id_val} not found in {table}"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/{table}")
async def create_record(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    id_val = payload.get("id") or payload.get("analysis_id")
    if not id_val:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must contain an 'id' or 'analysis_id'"
        )
    path = get_file_path(table, str(id_val))
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.patch("/{table}/{id_val}")
async def update_record(table: str, id_val: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = get_file_path(table, id_val)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    
    existing.update(payload)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        return existing
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{table}")
async def list_records(table: str):
    folder = os.path.join(BASE_DIR, table)
    if not os.path.exists(folder):
        return []
    records = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                pass
    return records

@router.delete("/{table}/{id_val}")
async def delete_record(table: str, id_val: str):
    path = get_file_path(table, id_val)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record {id_val} not found in {table}"
        )
    try:
        os.remove(path)
        return {"status": "success", "message": f"Deleted record {id_val}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
