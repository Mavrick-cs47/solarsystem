"""One error type for expected failures, rendered as a consistent JSON body."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("solarsight")


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, error: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Unexpected server error."}},
        )
