from typing import Annotated
from fastapi import FastAPI, File
from starlette.responses import StreamingResponse
from io import BytesIO
from service import Service, ModelType


service = Service()

app = FastAPI()


@app.post("/submit/")
async def submit(
    model: ModelType,
    image1: Annotated[bytes, File(description="Image 1")], image2: Annotated[bytes, File(description="Image 2")]
):
    fig = service.process(model, image1, image2)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")