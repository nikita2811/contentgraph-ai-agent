from fastapi import FastAPI, HTTPException,Depends,Header
from .agentstate import build_pipeline
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import traceback
from auth.jwt_verify import verify_service_token

app = FastAPI(
    title="AI Content Pipeline",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




# ── Models ────────────────────────────────────────────────

class ProductRequest(BaseModel):
    product_name:str
    key_features: list[str]
    category:str
    tone: str = "professional"
    target_audience: str = ""


class PipelineResponse(BaseModel):
    product_name:   str
    final_content:  Optional[str]
    research:       Optional[str]
    serp:           Optional[str]
    status:         str



# ── Routes ────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}




@app.post("/generate", dependencies=[Depends(verify_service_token)])
async def generate(req:ProductRequest):
   
     try:
        # 2. Prepare the initial state
      #    product_details = {
      #       "product_name": "Running Shoes",
      #       "category": "ecom",
      #       "target_audience": 20-40,
      #       "key_features": ["comfortable"],
      #       "tone": "professional"
      #   }
         product_details=req.model_dump()
    
         result = build_pipeline(product_details)
 
         return PipelineResponse(
            product_name  = product_details.get('product_name',""),
            final_content = result.get("final_content"),
            research      = result.get("raw_content"),
            serp          = result.get("serp_output"),
            status        = "success",
        )
     except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }
        )
        

     