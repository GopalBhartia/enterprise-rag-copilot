from fastapi import APIRouter
from pydantic import BaseModel

from app.services.feedback import save_feedback

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


class FeedbackRequest(BaseModel):
    trace_id: str
    query: str
    rating: str


@router.post("")
def submit_feedback(request: FeedbackRequest):

    save_feedback(request.model_dump())

    return {
        "status": "success",
        "message": "Feedback received",
    }
