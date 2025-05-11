#!/usr/bin/env python3
import argparse
import os

import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware

# Lấy alias cho backend và các API bạn cần
K = tf.keras.backend
load_model = tf.keras.models.load_model
Model      = tf.keras.models.Model

Input        = tf.keras.layers.Input
Conv2D       = tf.keras.layers.Conv2D
MaxPool2D    = tf.keras.layers.MaxPool2D
Activation   = tf.keras.layers.Activation
BatchNormalization    = tf.keras.layers.BatchNormalization
Add          = tf.keras.layers.Add
Lambda       = tf.keras.layers.Lambda
Bidirectional= tf.keras.layers.Bidirectional
LSTM         = tf.keras.layers.LSTM
Dense        = tf.keras.layers.Dense

# Hard-coded character list (from provided char_str)
char_str = " #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
char_list = list(char_str)
num_classes = len(char_list) +1 # +1 for CTC blank

IMAGE_HEIGHT = 118   # must match training height
IMAGE_WIDTH  = 2167  # must match training padded width
app = FastAPI()

# Configure CORS to allow requests from any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_crnn_model(img_h, num_classes):
    """Build CRNN model with dynamic-width input"""
    inputs = Input(shape=(img_h, None, 1), name='input')
    # Block1
    x = Conv2D(64,(3,3),padding='same')(inputs)
    x = MaxPool2D(pool_size=3,strides=3)(x)
    x = Activation('relu')(x)
    # Block2
    x = Conv2D(128,(3,3),padding='same')(x)
    x = MaxPool2D(pool_size=3,strides=3)(x)
    x = Activation('relu')(x)
    # Block3
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip3 = x
    # Block4
    x = Conv2D(256,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip3])
    x = Activation('relu')(x)
    # Block5
    x = Conv2D(512,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip5 = x
    # Block6
    x = Conv2D(512,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip5])
    x = Activation('relu')(x)
    # Block7
    x = Conv2D(1024,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,1))(x)
    x = Activation('relu')(x)
    # Pool to height=1
    x = MaxPool2D(pool_size=(3,1))(x)
    # squeeze height dimension: (batch, 1, T, features) -> (batch, T, features)
    x = Lambda(lambda t: K.squeeze(t, 1))(x)
    # BiLSTM layers
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(x)
    # softmax output
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='crnn_inference')


def preprocess_image(img_path, img_h):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_w = int(img_h / h * w)
    img = cv2.resize(img, (new_w, img_h))
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    debug_img = (img[0,:,:,0]*255).astype(np.uint8)
    cv2.imwrite(os.path.join('debug','preprocessed.png'), debug_img)
    return img

def segment_lines(binary_img, min_height=10, pad=10):
    """Segment binary image into lines with improved diacritic preservation"""
    h, w = binary_img.shape
    # 1. Vertical dilation to connect diacritics
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, pad))
    dilated = cv2.dilate(binary_img, vert_kernel, iterations=1)
    # 2. Horizontal closing to merge broken strokes
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 100,1), 1))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, horiz_kernel)
    # 3. Horizontal projection
    proj = np.sum(closed > 0, axis=1)
    thresh = proj.max() * 0.05
    lines = []
    in_line = False
    start = 0
    for y, v in enumerate(proj):
        if not in_line and v > thresh:
            in_line = True
            start = y
        elif in_line and v <= thresh:
            end = y
            if end - start >= min_height:
                y0 = max(0, start - pad)
                y1 = min(h, end + pad)
                lines.append((y0, y1))
            in_line = False
    # last line
    if in_line and h - start >= min_height:
        y0 = max(0, start - pad)
        y1 = h
        lines.append((y0, y1))
    return lines
def ctc_decode(pred, char_list):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True)
    seq = decoded[0].numpy()[0]
    return ''.join(char_list[i] for i in seq if i != -1)

IMAGE_DIR   = "images"
WEIGHTS_PATH = "model/checkpoint_weights.hdf5"

DEBUG_DIR="debug"
# Build model and load weights
print("Building model...")
model = build_crnn_model(IMAGE_HEIGHT, num_classes)
print(f"Loading weights from {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH)

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    # Read image bytes
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Threshold and debug save
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )
    cv2.imwrite(os.path.join(DEBUG_DIR, 'threshold.png'), th)

    # Segment lines and OCR
    lines = segment_lines(th)
    results = []
    for idx, (y0, y1) in enumerate(lines):
        crop = th[y0:y1, :]
        h, w = crop.shape
        new_w = int(IMAGE_HEIGHT / h * w)
        resized = cv2.resize(crop, (new_w, IMAGE_HEIGHT))
        seg_path = os.path.join(DEBUG_DIR, f'line_{idx+1}.png')
        cv2.imwrite(seg_path, resized)

        inp = resized.astype('float32') / 255.0
        inp = np.expand_dims(inp, axis=-1)[None]
        pred = model.predict(inp)
        text = ctc_decode(pred, char_list)
        results.append(text)


    return JSONResponse({"text": results})

if __name__ == "__main__":
    import uvicorn
    # Give Uvicorn the import path "app:app" instead of the instance
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )



