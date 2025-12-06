"""
FastAPI service for BEP text generation

Provides REST API endpoints for AI-assisted text generation in BEP documents.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from pathlib import Path

from model_loader import get_generator

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


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


# Initialize generator on startup
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        logger.info("Loading BEP text generation model...")
        generator = get_generator()
        logger.info(f"Model loaded successfully on device: {generator.device}")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Please train the model first using: python scripts/train_model.py")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


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
    try:
        generator = get_generator()
        return HealthResponse(
            status="healthy",
            model_loaded=generator.model is not None,
            device=str(generator.device)
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown"
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
    Generate field-specific suggestions

    Provides context-aware text suggestions for specific BEP fields.
    This endpoint is optimized for inline text completion in the BEP editor.

    - **field_type**: Type of field (e.g., 'executiveSummary', 'projectObjectives')
    - **partial_text**: Any text the user has already typed
    - **max_length**: Maximum characters to generate
    """
    try:
        generator = get_generator()

        if generator.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )

        # Generate field-specific suggestion
        suggestion = generator.suggest_for_field(
            field_type=request.field_type,
            partial_text=request.partial_text,
            max_length=request.max_length
        )

        logger.info(f"Generated suggestion for {request.field_type}: {len(suggestion)} chars")

        return GenerateResponse(
            text=suggestion,
            prompt_used=request.partial_text or generator.field_prompts.get(request.field_type, '')
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
