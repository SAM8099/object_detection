from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from starlette.responses import RedirectResponse
import tempfile
from test import counter  # import your counter() function

app = FastAPI(title="People Counter API",
              description="Counts number of people passing through a door using computer vision.",
              version="1.0")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
@app.post("/count")
async def count_people(video: UploadFile = File(...)):
    """
    Upload a video file and get the total number of people who passed through the door.
    """
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            contents = await video.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Call your counting function
        total_count = counter(temp_path)

        return JSONResponse(content={
            "status": "success",
            "video_name": video.filename,
            "people_passed": total_count
        })

    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "error": str(e)
        }, status_code=500)



