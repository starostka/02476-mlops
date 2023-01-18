from pydantic import BaseModel, Field
# from typing import Optional, List, Dict, Any


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    document: dict = Field(..., title='Model State Dictionary')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    pass


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Wheter there is an error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Wheter there is an error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(..., example='', title='Detailed traceback of the error')
