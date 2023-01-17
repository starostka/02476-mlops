#!/usr/bin/env python3
import torch
import uvicorn
import omegaconf
import asyncio
import functools
import concurrent.futures

from pipeline import Pipeline

from fastapi import FastAPI, UploadFile, File, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schema import *

from exception_handler import validation_exception_handler, python_exception_handler
from http import HTTPStatus

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FASTAPIInstrumentor
from opentelemetry.sdk.trace import TraceProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Set up telemetry for FastAPI
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

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
    license_info="MIT License"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.mount("/static", StaticFiles(directory="static/"), name="static")
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running environment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = GCN()
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "scaler": load(CONFIG['SCALAR_PATH']),  # joblib.load
        "model": model
    }


@app.get('/info')
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
        "torch.version.cuda": torch.version.cuda
    }


@app.post('/api/v1/predict', response_model=InferenceResponse, responses={422: {'model': ErrorResponse}, 500: {'model': ErrorResponse}})
def predict(request: Request, body: InferenceInput):
    """
    Perform prediction from data
    """

    # Monitoring: logs and telemetry
    logger.info('API predict called')
    logger.info(f"input: {body}")
    current_span = trace.get_current_span()

    # Define subroutines for prediction steps
    validate_data = partial(model.validate, *args)
    preprocess_data = partial(model.process, *args)
    do_predict = partial(model.predict, app.package, [X])

    # Initiate the request prediction
    prediction = do_predict()

    results = {
        'pred': prediction
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK
    }
    logger.info(f"results: {results}")
    return {'error': False, 'results': results}


@app.post('/feedback')
def receive_feedback(request):
    current_span = trace.get_current_span()
    save_to_db(request.feedback)
    current_span.set_attribute("app.demo.feedback", request.feedback)
    return {'received': 'ok'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True, debug=True, log_config="log.ini")
