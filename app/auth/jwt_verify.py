# auth/jwt_verifier.py
import jwt
import os
from functools import lru_cache
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
  
)
import base64
import os
from dotenv import load_dotenv,find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


bearer_scheme = HTTPBearer()
@lru_cache(maxsize=1)
def _get_public_key():
     raw = os.getenv("SERVICE_JWT_PUBLIC_KEY")
     if not raw:
        raise ValueError("SERVICE_JWT_PUBLIC_KEY env var is not set!")
     
    # Strip any accidental whitespace, quotes, or CRLF Windows added
     raw = raw.strip().strip('"').strip("'")
    
     try:
        pem_bytes = base64.b64decode(raw)           # decode base64 → raw PEM bytes
     except Exception as e:
        raise ValueError(f"Base64 decode failed: {e}\nRaw value start: {raw[:50]}")
    
    
     try:
        return load_pem_public_key(pem_bytes)
     except Exception as e:
        # Print decoded PEM to help debug framing issues
        raise ValueError(f"PEM load failed: {e}\nDecoded PEM:\n{pem_bytes.decode()}")

def verify_service_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    
    
    token = credentials.credentials


    try:
        payload = jwt.decode(
            token,
            _get_public_key(),
            algorithms=["RS256"],
            audience="fastapi-service",       # must match what Django set
            options={"require": ["exp", "iss", "aud", "iat"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid audience")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid issuer")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    # Extra guard — only accept tokens from known service identities
    allowed_issuers = {"django-service"}
    if payload.get("iss") not in allowed_issuers:
        raise HTTPException(status_code=403, detail="Untrusted issuer")

    return payload