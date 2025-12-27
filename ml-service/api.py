"""
FastAPI service for BEP text generation

Provides REST API endpoints for AI-assisted text generation in BEP documents.
Supports both RAG (Retrieval-Augmented Generation) with Claude API and
LSTM-based fallback generation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
from pathlib import Path
import os

from model_loader import get_generator
from rag_engine import get_rag_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BEP AI Text Generator",
    description="AI-powered text generation for BIM Execution Plans (ISO 19650)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Starting text prompt")
    field_type: Optional[str] = Field(None, description="Type of BEP field")
    max_length: int = Field(200, ge=50, le=1000, description="Maximum characters to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    text: str = Field(..., description="Generated text")
    prompt_used: str = Field(..., description="Actual prompt used for generation")
    method: str = Field(default="lstm", description="Generation method used (rag or lstm)")
    sources: Optional[List[Dict]] = Field(None, description="Source documents (RAG only)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    lstm_model_loaded: bool
    device: str
    rag_available: bool
    rag_status: Optional[Dict] = None


# Initialize generator on startup
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    # Load LSTM model
    try:
        logger.info("Loading LSTM text generation model...")
        generator = get_generator()
        logger.info(f"LSTM model loaded successfully on device: {generator.device}")
    except FileNotFoundError as e:
        logger.error(f"LSTM model not found: {e}")
        logger.error("Please train the model first using: python scripts/train_model.py")
    except Exception as e:
        logger.error(f"Error loading LSTM model: {e}")

    # Load RAG engine
    try:
        logger.info("Loading RAG engine...")
        rag = get_rag_engine()
        status = rag.get_status()
        if status['llm_available']:
            logger.info("RAG engine loaded successfully with Claude API")
        else:
            logger.warning("RAG engine loaded but Claude API not configured")
    except Exception as e:
        logger.error(f"Error loading RAG engine: {e}")
        logger.info("RAG features will be unavailable, falling back to LSTM only")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "BEP AI Text Generator API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    lstm_loaded = False
    device = "unknown"
    rag_available = False
    rag_status = None

    # Check LSTM
    try:
        generator = get_generator()
        lstm_loaded = generator.model is not None
        device = str(generator.device)
    except Exception as e:
        logger.error(f"LSTM health check error: {e}")

    # Check RAG
    try:
        rag = get_rag_engine()
        rag_status = rag.get_status()
        rag_available = rag_status['llm_available'] and rag_status['vectorstore_loaded']
    except Exception as e:
        logger.error(f"RAG health check error: {e}")
        rag_status = {"error": str(e)}

    overall_status = "healthy" if (lstm_loaded or rag_available) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        lstm_model_loaded=lstm_loaded,
        device=device,
        rag_available=rag_available,
        rag_status=rag_status
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    """
    Generate text based on a prompt

    This endpoint uses a trained LSTM model to generate contextually appropriate
    text for BIM Execution Plan documents.

    - **prompt**: Starting text to continue from
    - **field_type**: Optional field type for context-specific generation
    - **max_length**: Maximum characters to generate (50-1000)
    - **temperature**: Sampling temperature (0.1-2.0, higher = more creative)
    """
    try:
        generator = get_generator()

        if generator.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )

        # Generate text
        if request.field_type:
            # Use field-specific generation
            generated = generator.suggest_for_field(
                field_type=request.field_type,
                partial_text=request.prompt,
                max_length=request.max_length
            )
            prompt_used = request.prompt
        else:
            # Use generic prompt-based generation
            full_text = generator.generate_text(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            generated = full_text[len(request.prompt):]
            prompt_used = request.prompt

        logger.info(f"Generated {len(generated)} characters for field_type: {request.field_type}")

        return GenerateResponse(
            text=generated.strip(),
            prompt_used=prompt_used
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model files not found. Please train the model first using: python scripts/train_model.py"
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


class SuggestRequest(BaseModel):
    """Request model for field suggestions"""
    field_type: str = Field(..., description="Type of BEP field")
    partial_text: str = Field("", description="Existing text in the field")
    max_length: int = Field(200, ge=50, le=1000, description="Maximum characters to generate")


@app.post("/suggest", response_model=GenerateResponse, tags=["Generation"])
async def suggest_for_field(request: SuggestRequest):
    """
    Generate field-specific suggestions with RAG

    Provides context-aware text suggestions for specific BEP fields.
    Uses RAG (Retrieval-Augmented Generation) with Claude API when available,
    falls back to LSTM model if RAG is unavailable.

    - **field_type**: Type of field (e.g., 'executiveSummary', 'projectObjectives')
    - **partial_text**: Any text the user has already typed
    - **max_length**: Maximum characters to generate

    Returns:
    - **text**: Generated text
    - **method**: Generation method used ('rag' or 'lstm')
    - **sources**: Source documents (only for RAG method)
    """
    # Try RAG first
    try:
        rag = get_rag_engine()
        rag_status = rag.get_status()

        if rag_status['llm_available'] and rag_status['vectorstore_loaded']:
            logger.info(f"Generating with RAG for field: {request.field_type}")

            result = rag.generate_suggestion(
                field_type=request.field_type,
                partial_text=request.partial_text,
                max_length=request.max_length,
                k=3
            )

            logger.info(f"RAG generated {len(result['text'])} chars from {result['retrieved_chunks']} chunks")

            return GenerateResponse(
                text=result['text'],
                prompt_used=request.partial_text,
                method="rag",
                sources=result['sources']
            )

    except Exception as e:
        logger.warning(f"RAG generation failed, falling back to LSTM: {e}")

    # Fallback to LSTM
    try:
        logger.info(f"Using LSTM fallback for field: {request.field_type}")
        generator = get_generator()

        if generator.model is None:
            raise HTTPException(
                status_code=503,
                detail="Both RAG and LSTM models unavailable. Please configure API key or train LSTM model."
            )

        # Generate field-specific suggestion with LSTM
        suggestion = generator.suggest_for_field(
            field_type=request.field_type,
            partial_text=request.partial_text,
            max_length=request.max_length
        )

        logger.info(f"LSTM generated {len(suggestion)} chars for {request.field_type}")

        return GenerateResponse(
            text=suggestion,
            prompt_used=request.partial_text or generator.field_prompts.get(request.field_type, ''),
            method="lstm",
            sources=None
        )

    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Suggestion generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    print("="*60)
    print("Starting BEP AI Text Generator API")
    print("="*60)
    print("API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
