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
        
        
        result = await run_pipeline(product_details)
        if result.get("error"):
           raise ValueError(f"Pipeline failed at step '{result.get('current_step')}': {result['error']}")

      
        content = result.get("content_output") or {}
        serp = result.get("serp_output") or {}
        # Handle tuple returned from DB/fetchone(), just in case
        if isinstance(content, tuple):
         content = content[0]
        if isinstance(serp, tuple):
            serp = serp[0]
        
        # If somehow still a JSON string, parse it
        if isinstance(content, str):
            content = json.loads(content)
        if isinstance(serp, str):
            serp = json.loads(serp)
        
        ai_content = content
        serp_final = serp


        return JSONResponse(content={
            "product_name":  product_details.get("product_name", ""),
            "seo_title": ai_content["seo_title"],
            "meta_description": ai_content["meta_description"],
            "meta_title": ai_content["h1"],
            "intro_paragraph": ai_content["intro_paragraph"],
            "tags": ai_content["tags"] if isinstance(ai_content["tags"], list) else ai_content["tags"].split(","),
            "primary_keyword": serp_final["primary_keyword"],
            "secondary_keyword": ",".join(serp_final["secondary_keywords"]) if isinstance(serp_final["secondary_keywords"], list) else serp["secondary_keywords"],
            "status":        "success",
            "token_usage":   result.get("token_usage", {}),
           
           
        })


    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
             detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),  # ← full traceback in response
            }
            # detail={"error": str(e)}  # ✅ never expose traceback to clients
        )