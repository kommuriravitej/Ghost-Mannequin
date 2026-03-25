import os
import io
from fastapi import FastAPI, File, UploadFile
import replicate
from rembg import remove

app = FastAPI()

@app.post("/process-ghost")
async def process_ghost(garment: UploadFile = File(...)):
    # 1. Remove table/floor background
    input_bytes = await garment.read()
    clean_garment = remove(input_bytes)
    
    # 2. Trigger the AI (Invisible Mannequin Effect)
    # Note: REPLICATE_API_TOKEN must be set in your Server Variables
    prediction = replicate.run(
        "cuuupid/idm-vton:8830e9d6",
        input={
            "garm_img": io.BytesIO(clean_garment),
            "human_img": "https://replicate.delivery/pbxt/L1S/ghost_template.jpg",
            "garment_des": "ghost mannequin, 3d hollow neck, baby clothes",
            "category": "dresses" 
        }
    )
    return {"ghost_url": prediction[0]}
