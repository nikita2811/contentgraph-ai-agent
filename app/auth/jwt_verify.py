# auth/jwt_verifier.py
import jwt
import os
from functools import lru_cache
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


bearer_scheme = HTTPBearer()

@lru_cache(maxsize=1)
def _get_public_key():
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    key_bytes = os.environ["SERVICE_JWT_PUBLIC_KEY"].encode()
    return load_pem_public_key(key_bytes)

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