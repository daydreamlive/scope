"""PersonaLive-specific API routes.

This module defines custom API endpoints for the PersonaLive pipeline,
registered via the register_routes hook.
"""

import logging
from io import BytesIO

from fastapi import Depends, FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PersonaLiveReferenceResponse(BaseModel):
    """Response after setting PersonaLive reference image."""

    success: bool = Field(
        ..., description="Whether the reference image was set successfully"
    )
    message: str = Field(..., description="Status message")


def register_personalive_routes(app: FastAPI):
    """Register PersonaLive-specific routes with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Import here to avoid circular imports
    from scope.server.app import get_pipeline_manager
    from scope.server.pipeline_manager import PipelineManager

    @app.post(
        "/api/v1/personalive/reference", response_model=PersonaLiveReferenceResponse
    )
    async def set_personalive_reference(
        request: Request,
        pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
    ):
        """Set the reference image for PersonaLive pipeline.

        This endpoint accepts a reference portrait image that will be animated
        based on driving video frames. The image should be a face-centered portrait.

        The request body should be the raw image bytes with appropriate content-type
        header (image/jpeg, image/png, etc.).

        This must be called after loading the PersonaLive pipeline and before
        starting video streaming.
        """
        try:
            # Check if PersonaLive pipeline is loaded
            status_info = await pipeline_manager.get_status_info_async()
            if status_info.get("pipeline_id") != "personalive":
                raise HTTPException(
                    status_code=400, detail="PersonaLive pipeline must be loaded first"
                )
            if status_info.get("status") != "loaded":
                raise HTTPException(
                    status_code=400,
                    detail=f"Pipeline not ready. Current status: {status_info.get('status')}",
                )

            # Get image from request body
            body = await request.body()
            if not body:
                raise HTTPException(status_code=400, detail="No image data provided")

            # Load image
            try:
                image = Image.open(BytesIO(body)).convert("RGB")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image data: {e}"
                ) from e

            # Get pipeline and fuse reference
            pipeline = pipeline_manager.get_pipeline()
            if pipeline is None:
                raise HTTPException(status_code=500, detail="Pipeline not available")

            # Fuse reference image
            pipeline.fuse_reference(image)

            return PersonaLiveReferenceResponse(
                success=True,
                message="Reference image set successfully. Ready to process driving video.",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error setting PersonaLive reference: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
