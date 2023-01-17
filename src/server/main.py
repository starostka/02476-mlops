#!/usr/bin/env python3
import os
import sys
import torch
import joblib
import uvicorn
import omegaconf
import asyncio
import concurrent.futures

from src.models.pipeline import Pipeline
from functools import partial

from fastapi import FastAPI, UploadFile, File, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schema import InferenceInput, InferenceResponse, InferenceResult, ErrorResponse
from http import HTTPStatus

# from opentelemetry import trace
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.fastapi import FASTAPIInstrumentor
# from opentelemetry.sdk.trace import TraceProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor

# FIXME Set up telemetry for FastAPI
# provider = TraceProvider()
# processor = BatchSpanProcessor(OTLPSpanExporter())
# provider.add_span_processor(processor)
# trace.set_tracer_provider(provider)
# tracer = trace.get_tracer(__name__)

# Set up the FastAPI app/service
# Inspiration:
# - Slides from Duarts
# - Medium tutorial: https://medium.com/@mingc.me/deploying-pytorch-model-to-production-with-fastapi-in-cuda-supported-docker-c161cca68bb8
app = FastAPI(
    title="MLOps API",
    description="Example API for the deployed model.",
    version="0.0.1",
    terms_of_service=None,
    contact="benjamin@starostka.io",
    license_info="MIT License",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.mount("/static", StaticFiles(directory="static/"), name="static")


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    # Load pipeline for the served model
    global pipeline
    pipeline = Pipeline()

    logger.info("Running environment: {}".format(pipeline.config["ENV"]))
    logger.info("PyTorch using device: {}".format(pipeline.config["DEVICE"]))

    # FIXME add model and other preprocess tools to app state
    # app.package = {
    #     # "scaler": joblib.load(pipeline.config['SCALAR_PATH']),
    #     "pipeline": pipeline
    # }


@app.get("/info")
def show_info():
    """
    Show server and system information as a health check.
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
    }


@app.post(
    "/api/v1/predict",
    response_model=InferenceResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict(request: Request, body: InferenceInput):
    """
    Perform prediction from data
    """

    # Monitoring: logs and telemetry
    logger.info("API predict called")
    logger.info(f"input: {body}")
    # FIXME current_span = trace.get_current_span()

    # FIXME Trigger the model predict for the request
    prediction = pipeline.model.predict(body.data)

    results = {
        "pred": prediction,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    logger.info(f"results: {results}")
    return {"error": False, "results": results}


@app.post("/feedback")
def receive_feedback(request):
    # FIXME current_span = trace.get_current_span()
    # FIXME save_to_db(request.feedback)
    # FIXME current_span.set_attribute("app.demo.feedback", request.feedback)
    return {"received": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        debug=True,
        log_config="log.ini",
    )
