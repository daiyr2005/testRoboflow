from fastapi import APIRouter, HTTPException, File, UploadFile

from predict import get_prediction

router = APIRouter(prefix='/api', tags=['CheckAnimal'])

@router.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail='Файл туура эмес')

        result = await get_prediction(image_bytes)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))