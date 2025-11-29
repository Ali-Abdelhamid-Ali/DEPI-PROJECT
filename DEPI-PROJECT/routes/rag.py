from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import logging
import os
import shutil
from pathlib import Path

from .schemas.rag import (
    RAGQueryRequest,
    DocumentIndexRequest,
    DocumentSearchRequest,
    DocumentDeleteRequest,
    DocumentListRequest,
    DocumentUploadResponse
)
from helpers.configs import Settings, get_settings
from controllers.RAGController import RAGController

logger = logging.getLogger('uvicorn.error')

rag_router = APIRouter(
    prefix=f"/{get_settings().APP_NAME}/rag",
    tags=["RAG"]
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@rag_router.post("/query")
async def rag_query(request: RAGQueryRequest, app_settings: Settings = Depends(get_settings)):
    """
    RAG Query endpoint: retrieve relevant context and generate answer
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )

        response = await controller.rag_query(
            query=request.query,
            top_k=request.top_k
        )

        return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "query": request.query,
            "response": response
        })

    except Exception as e:
        logger.exception("RAG query error:")
        raise HTTPException(status_code=500, detail=str(e))
    
@rag_router.post("/search")
async def search_documents(request: DocumentSearchRequest, app_settings: Settings = Depends(get_settings)):
    """
    Search for relevant documents without generating answer
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        results = await controller.search_documents(
            query=request.query,
            top_k=request.top_k
        )

        return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "query": request.query,
            "results": results,
            "num_results": len(results)
        })

    except Exception as e:
        logger.exception("Search documents error:")
        raise HTTPException(status_code=500, detail=str(e))



@rag_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = None,
    username: str = None
):
    """
    Upload a document file
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = ['.txt', '.pdf', '.docx', '.md']

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Create user-specific directory
        user_dir = os.path.join(UPLOAD_DIR, username or "default")
        os.makedirs(user_dir, exist_ok=True)

        # Save file
        file_path = os.path.join(user_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse({
            "success": True,
            "file_name": file.filename,
            "file_path": file_path,
            "message": f"File {file.filename} uploaded successfully"
        })

    except Exception as e:
        logger.exception("Upload document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/index")
async def index_document(request: DocumentIndexRequest, app_settings: Settings = Depends(get_settings)):
    """
    Index an uploaded document
    """
    try:
        # Construct file path
        user_dir = os.path.join(UPLOAD_DIR, request.username or "default")
        file_path = os.path.join(user_dir, request.file_name)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_name}")

        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )

        result = await controller.index_document(file_path)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Index document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/index-directory")
async def index_directory(request: DocumentIndexRequest, app_settings: Settings = Depends(get_settings)):
    """
    Index all documents in user's upload directory
    """
    try:
        user_dir = os.path.join(UPLOAD_DIR, request.username or "default")

        if not os.path.exists(user_dir):
            raise HTTPException(status_code=404, detail=f"Directory not found for user: {request.username}")

        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )

        result = await controller.index_directory(user_dir)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Index directory error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/delete")
async def delete_document(request: DocumentDeleteRequest, app_settings: Settings = Depends(get_settings)):
    """
    Delete a document from the vector store
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        result = await controller.delete_document(request.doc_id)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Delete document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/list")
async def list_documents(request: DocumentListRequest, app_settings: Settings = Depends(get_settings)):
    """
    List documents in the vector store
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        result = await controller.list_documents(
            limit=request.limit,
            offset=request.offset
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("List documents error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.get("/stats")
async def get_statistics(app_settings: Settings = Depends(get_settings)):
    """
    Get RAG system statistics
    """
    try:
        controller = RAGController()
        result = await controller.get_statistics()

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Get statistics error:")
        raise HTTPException(status_code=500, detail=str(e))
