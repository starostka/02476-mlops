from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    document_network: dict = Field(..., title='Model State Dictionary')

class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    pass

class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    pass

class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    pass
