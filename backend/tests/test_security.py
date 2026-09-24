"""Tests for security and token utilities."""
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token
)


def test_password_hashing():
    raw_password = "GovSecure#2026"
    hashed = get_password_hash(raw_password)
    assert hashed != raw_password
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword#123", hashed) is False


def test_jwt_token_lifecycle():
    subject = "officer-GOV-IN-7741"
    token = create_access_token(subject, extra_claims={"role": "Authorized Commander"})
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == subject
    assert payload["role"] == "Authorized Commander"


def test_invalid_jwt_token():
    assert decode_access_token("invalid.token.structure") is None
