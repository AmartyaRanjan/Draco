import re

from app.agent.contracts import CapabilityError, ErrorCode


def require(service, name: str, hint: str = ""):
    if service is None:
        raise CapabilityError(ErrorCode.not_configured, f"{name} is not configured. {hint}".strip())
    return service


def check_scope(ctx, document_id: str):
    if ctx.document_scope and document_id not in ctx.document_scope:
        raise CapabilityError(ErrorCode.denied, "Document is outside the permitted scope.")


_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def safe_id(value: str) -> str:
    if not _ID.match(value):
        raise CapabilityError(ErrorCode.invalid_input, "Malformed identifier.")
    return value
