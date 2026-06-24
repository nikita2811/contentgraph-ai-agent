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
        content = result.get("content_output") or ""
        serp = result.get("serp_output") or ""


        final_content = content if isinstance(content, str) else json.dumps(content),
        ai_serp = serp if isinstance(serp, str) else json.dumps(serp),

      

        # fetchone() returns tuple
        if isinstance(final_content, tuple):
            final_content = final_content[0]
        
        # JSON string -> Python object
        if isinstance(final_content, str):
            content_list = json.loads(final_content)
        else:
            content_list = final_content
        
        print(type(content_list))
        print(content_list)
        
        content_text = next(
            (
                block["text"]
                for block in content_list
                if isinstance(block, dict)
                and block.get("type") == "text"
            ),
            None
        )
        
        if not content_text:
            raise ValueError("No text block found in final_content")
        
        content_text = (
            content_text
            .strip()
            .removeprefix("```json")
            .removesuffix("```")
            .strip()
        )
        
        ai_content = json.loads(content_text)
        

        # Handle tuple returned from DB/fetchone()
        if isinstance(ai_serp, tuple):
            ai_serp = ai_serp[0]
        
        # Parse JSON string
        if isinstance(ai_serp, str):
            serp_raw = json.loads(ai_serp)
        else:
            serp_raw = ai_serp
        
        # Extract text block
        serp_text = next(
            (
                block["text"]
                for block in serp_raw
                if isinstance(block, dict)
                and block.get("type") == "text"
            ),
            None
        )
        
        if not serp_text:
            raise ValueError("No text block found in serp")
        
        # Remove markdown fences if present
        serp_text = (
            serp_text
            .strip()
            .removeprefix("```json")
            .removesuffix("```")
            .strip()
        )
        
        # Convert AI JSON string to Python dict
        serp_final = json.loads(serp_text)


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
        raise HTTPException(
            status_code=500,
             detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),  # ← full traceback in response
            }
            # detail={"error": str(e)}  # ✅ never expose traceback to clients
        )