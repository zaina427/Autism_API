"""
Autism Detection FastAPI
Serves the autism detection model as a REST API
"""

import os
import io
import uuid
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager

# Create FastAPI app
app = FastAPI(
    title="Autism Detection API",
    description="API for detecting autism characteristics from facial images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Global model variable
model = None

# Model configuration
MODEL_INPUT_SIZE = (224, 224)  # Model expects 224x224 images
MODEL_THRESHOLD = 0.5  # Classification threshold

def load_model():
    global model
    model_path = "autism_detection_model.h5"
    print(f"Loading model from {model_path}...")

    try:
        # Most reliable way for .h5 files
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: Could not load model. {e}")
        print("TIP: Ensure your environment has the same Keras/TensorFlow version used during training.")

    
from keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img = cv2.resize(img, MODEL_INPUT_SIZE)

        # Convert to float32
        img = img.astype(np.float32)

        # MobileNetV2 preprocessing
        img = preprocess_input(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def save_uploaded_file(file_data: bytes, filename: str) -> str:
    """
    Save uploaded file temporarily
    
    Args:
        file_data: File bytes
        filename: Original filename
        
    Returns:
        Saved file path
    """
    try:
        # Generate unique filename
        file_ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join("uploads", unique_filename)
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(file_data)
        
        return filepath
    except Exception as e:
        print(f" Error saving file: {e}")
        return None

def cleanup_file(filepath: str):
    """
    Delete temporary file
    
    Args:
        filepath: Path to file to delete
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f" Error cleaning up file: {e}")

# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    print("Starting Autism Detection API...")
    load_model()
    yield
    # Clean up (if needed) on shutdown
    print("Shutting down...")

# Remove the old @app.on_event("startup") block entirely

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": " Autism Detection API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Check API health",
            "GET /model-info": "Get model information",
            "POST /predict": "Upload image for prediction",
            "GET /docs": "Interactive API documentation"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model": {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "input_size": MODEL_INPUT_SIZE,
            "threshold": MODEL_THRESHOLD
        },
        "preprocessing": {
            "resize": MODEL_INPUT_SIZE,
            "normalization": "0 to 1 (divide by 255)",
            "color_format": "RGB"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image file to analyze"),
    save_temp: bool = False
):
    """
    Predict autism characteristics from an uploaded image
    
    - **image**: Image file (JPEG, PNG, JPG)
    - **save_temp**: Whether to save the uploaded file temporarily (for debugging)
    
    Returns prediction results
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Read image
        contents = await image.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size: 10MB")
        
        # Save file temporarily if requested
        temp_filepath = None
        if save_temp:
            temp_filepath = save_uploaded_file(contents, image.filename)
            if temp_filepath:
                background_tasks.add_task(cleanup_file, temp_filepath)
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess image
        processed_image = preprocess_image(contents)
        
        # Make prediction
        start_time = datetime.now()
        prediction = model.predict(processed_image, verbose=0)
        print("Raw prediction:", prediction)
        print("Prediction shape:", prediction.shape)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Get probability
        probability = float(prediction[0][0])
        
        # Interpret results
        if probability > MODEL_THRESHOLD:
            label = "Autistic"
            confidence = probability
        else:
            label = "Non-autistic"
            confidence = 1 - probability
        
        # Format response
        response = {
            "success": True,
            "prediction": {
                "label": label,
                "probability": probability,
                "confidence": confidence,
                "threshold": MODEL_THRESHOLD,
                "interpretation": f"The model predicts {label.lower()} characteristics with {confidence:.1%} confidence."
            },
            "image_info": {
                "filename": image.filename,
                "content_type": image.content_type,
                "size_bytes": len(contents),
                "dimensions": MODEL_INPUT_SIZE
            },
            "performance": {
                "inference_time_seconds": round(inference_time, 4),
                "timestamp": datetime.now().isoformat()
            },
            "model_info": {
                "input_size": MODEL_INPUT_SIZE,
                "normalization": "0-1",
                "color_space": "RGB"
            }
        }
        
        # Add file path if saved
        if temp_filepath:
            response["image_info"]["temp_path"] = temp_filepath
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f" Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(
    images: list[UploadFile] = File(..., description="Multiple image files"),
    max_images: int = 5
):
    """
    Predict autism characteristics from multiple images
    
    - **images**: List of image files
    - **max_images**: Maximum number of images to process (default: 5)
    
    Returns batch prediction results
    """
    try:
        # Limit number of images
        if len(images) > max_images:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many images. Maximum allowed: {max_images}"
            )
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        processed_count = 0
        start_time = datetime.now()
        
        for image in images:
            try:
                # Validate file type
                allowed_types = ["image/jpeg", "image/png", "image/jpg"]
                if image.content_type not in allowed_types:
                    results.append({
                        "filename": image.filename,
                        "success": False,
                        "error": f"Invalid file type: {image.content_type}"
                    })
                    continue
                
                # Read image
                contents = await image.read()
                
                if len(contents) == 0:
                    results.append({
                        "filename": image.filename,
                        "success": False,
                        "error": "Empty file"
                    })
                    continue
                
                # Preprocess and predict
                processed_image = preprocess_image(contents)
                prediction = model.predict(processed_image, verbose=0)
                probability = float(prediction[0][0])
                
                # Interpret results
                if probability > MODEL_THRESHOLD:
                    label = "Autistic"
                    confidence = probability
                else:
                    label = "Non-autistic"
                    confidence = 1 - probability
                
                results.append({
                    "filename": image.filename,
                    "success": True,
                    "label": label,
                    "probability": probability,
                    "confidence": confidence,
                    "size_bytes": len(contents)
                })
                
                processed_count += 1
                
            except Exception as e:
                results.append({
                    "filename": image.filename,
                    "success": False,
                    "error": str(e)
                })
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "summary": {
                "total_images": len(images),
                "processed_successfully": processed_count,
                "failed": len(images) - processed_count,
                "total_time_seconds": round(total_time, 4)
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/test-prediction")
async def test_prediction():
    """
    Test endpoint with a sample prediction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_bytes = buffer.tobytes()
        
        # Preprocess
        processed_image = preprocess_image(image_bytes)
        
        # Predict
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        
        return {
            "success": True,
            "message": "Test prediction successful",
            "prediction": probability,
            "label": "Autistic" if probability > MODEL_THRESHOLD else "Non-autistic",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": request.url.path}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

# Run the application
if __name__ == "__main__":
    print(" Starting Autism Detection API server...")
    print(" API Documentation: http://localhost:8000/docs")
    print(" Model input size: 224x224")
    print("  Threshold: 0.5")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )