# app/main.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import traceback
import json

from .agentstate import run_pipeline
from app.auth.jwt_verify import verify_service_token
from .cache import init_llm_cache
from .token_usage import TokenUsageCallback


app = FastAPI(
    title="AI Content Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ─────────────────────────────────────────────────
class ProductRequest(BaseModel):
    product_name:    str
    key_features:    List[str]
    category:        str
    tone:            str = "professional"
    target_audience: str = ""
    regenerate:      bool = False  # ✅ new field


class PipelineResponse(BaseModel):
    product_name:  str
    final_content: Optional[str]
    serp:          Optional[str]
    status:        str
    token_usage:   Optional[dict] = None


# ── Init cache once at startup ─────────────────────────────
init_llm_cache(ttl=3600)

# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/generate", response_model=PipelineResponse, dependencies=[Depends(verify_service_token)])
async def generate(req: ProductRequest):
    product_details = req.model_dump()

    try:
        token_callback = TokenUsageCallback()
        result = await run_pipeline(product_details,callbacks=[token_callback])
       
        content = result.get("content_output") or ""
        serp = result.get("serp_output") or ""

        return JSONResponse(content={
            "product_name":  product_details.get("product_name", ""),
            "final_content": content if isinstance(content, str) else json.dumps(content),
            "serp":          serp if isinstance(serp, str) else json.dumps(serp),
            "status":        "success",
           
            "token_usage": {
              "prompt_tokens":     token_callback.prompt_tokens,
              "completion_tokens": token_callback.completion_tokens,
              "total_tokens":      token_callback.total_tokens,
              "model_name":        token_callback.model_name,
            }
        })


    except Exception as e:
        raise HTTPException(
            status_code=500,
             detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),  # ← full traceback in response
            }
            # detail={"error": str(e)}  # ✅ never expose traceback to clients
        )