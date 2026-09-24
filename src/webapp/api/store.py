from fastapi import APIRouter, Depends, HTTPException, status

from store import vector_store
from ..auth import basic_auth

router = APIRouter(prefix="/api/store", tags=["store"])

CONFIRMATION = "delete-the-index"


@router.post("/clean")
async def clean_store(confirm: str = "", credentials=Depends(basic_auth)):
    if confirm != CONFIRMATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This deletes the vector index. Repeat the request with ?confirm={CONFIRMATION}",
        )
    vector_store.clean()
