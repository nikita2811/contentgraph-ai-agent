# test_pipeline.py — run with: python test_pipeline.py
import asyncio
import traceback
from app.agentstate import build_pipeline, run_pipeline

async def test():
    try:
        pipeline = build_pipeline()
        print("✅ build_pipeline() OK")
    except Exception as e:
        print("❌ build_pipeline() FAILED")
        traceback.print_exc()
        return

    try:
        result = await run_pipeline({
            "product_name":    "EcoSip Steel Bottle",
            "category":        "drinkware",
            "target_audience": "25-40",
            "key_features":    ["insulated", "eco-friendly"],
            "tone":            "professional",
            "regenerate":      False,
        })
        print("✅ run_pipeline() OK")
        print(result)
    except Exception as e:
        print("❌ run_pipeline() FAILED")
        traceback.print_exc()

asyncio.run(test())