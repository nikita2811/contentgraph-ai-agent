from fastapi import FastAPI, HTTPException
from .agentstate import run_pipeline

app = FastAPI()


@app.post("/research")
async def research():
   
     try:
        # 2. Prepare the initial state
         product_details = {
            "product_name": "Running Shoes",
            "category": "ecom",
            "target_audience": 20-40,
            "key_features": ["comfortable"],
            "tone": "professional"
        }
    
         result = run_pipeline(product_details)
 
         print("\n\n━━━ FINAL ARTICLE ━━━\n")
         print(result["final_content"])
     
     
    
         return {
            result["final_content"]
        }
     except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

     