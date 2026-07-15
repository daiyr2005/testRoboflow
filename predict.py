import base64
import requests
from fastapi import HTTPException
from config import config  # Импортируем готовый конфиг


async def get_prediction(images_bytes: bytes):
    try:
        print('Roboflow: Отправка картинки...')

        # 1. Переводим байты изображения в Base64 строку (как просит Roboflow API)
        image_base64 = base64.b64encode(images_bytes).decode('utf-8')

        # 2. Формируем правильный URL вместе с API-ключом
        url = f"{config.ROBOFLOW_URL}?api_key={config.ROBOFLOW_API_KEY}"

        # 3. Отправляем запрос методом POST в формате urlencoded
        response = requests.post(
            url,
            data=image_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        print(f"Статус ответа: {response.status_code}")
        print(f"Текст ответа от Roboflow: {response.text}")

        # Если Roboflow ответил ошибкой, мы увидим её текст в Swagger
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Roboflow API Error {response.status_code}: {response.text}"
            )

        # 4. Возвращаем успешный JSON ответ от нейросети
        return {"inference_id": response.json()['inference_id'],
                "time": response.json()['time'],
                "class": response.json()['predictions'][0]['class'],
                "confidence": response.json()['predictions'][0]['confidence'] * 100}
        #
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ошипка, {e}')

































# import requests
# import io
# from fastapi import HTTPException
# from moto.efs.urls import response
#
# from config import config
#
#
# async def get_prediction(images_bytes: bytes):
#     try:
#         print('Roboflow')
#         url = config.ROBOFLOW_URL
#         files = {
#             'file': ('image.jpg', io.BytesIO(images_bytes), 'image/jpg')
#         }
#         response = requests.post(url, files=files)
#         print(response.text)
#
#         if response.status_code != 200:
#             raise HTTPException(status_code=500, detail='Ошипка сервер')
#
#         return {"confidence": response.json()}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'Ошипка, {e}')
