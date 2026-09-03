# test_pipeline.py — run with: python test_pipeline.py
import asyncio
import time
import traceback
from datetime import datetime
from app.agentstate import build_pipeline, run_pipeline


def fmt(seconds: float) -> str:
    """Format seconds as Xs or Xm Ys for readability."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.2f}s"


async def test():
    print(f"🕐 Test started at {datetime.now().strftime('%H:%M:%S')}")
    overall_start = time.monotonic()

    # ── build_pipeline() ──────────────────────────────────
    t0 = time.monotonic()
    try:
        pipeline = build_pipeline()
        build_time = time.monotonic() - t0
        print(f"✅ build_pipeline() OK — took {fmt(build_time)}")
    except Exception:
        print(f"❌ build_pipeline() FAILED — after {fmt(time.monotonic() - t0)}")
        traceback.print_exc()
        return

    # ── run_pipeline() ────────────────────────────────────
    t1 = time.monotonic()
    try:
        result = await run_pipeline({
            "product_name":    "EcoSip Steel Bottle",
            "category":        "drinkware",
            "target_audience": "25-40",
            "key_features":    ["insulated", "eco-friendly"],
            "tone":            "professional",
            "regenerate":      True,
        })
        run_time = time.monotonic() - t1
        print(f"✅ run_pipeline() OK — took {fmt(run_time)}")
        print(result)

        # Surface token usage if present — helps correlate slow calls with token volume
        usage = result.get("token_usage") if isinstance(result, dict) else None
        if usage:
            print(f"📊 Token usage: {usage}")

        retry_count = result.get("retry_count") if isinstance(result, dict) else None
        if retry_count:
            print(f"🔄 Rewrite loop fired {retry_count} time(s) — this adds significant latency per pass")

    except Exception:
        run_time = time.monotonic() - t1
        print(f"❌ run_pipeline() FAILED — after {fmt(run_time)}")
        traceback.print_exc()

    # ── Summary ───────────────────────────────────────────
    total_time = time.monotonic() - overall_start
    print("\n" + "=" * 50)
    print(f"⏱  build_pipeline(): {fmt(build_time)}")
    print(f"⏱  run_pipeline():   {fmt(run_time)}")
    print(f"⏱  TOTAL:            {fmt(total_time)}")
    print("=" * 50)


asyncio.run(test())