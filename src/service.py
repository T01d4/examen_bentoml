import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from fastapi import HTTPException
from jose import JWTError, jwt
from datetime import datetime, timedelta
import numpy as np

# Secret key for JWT encoding/decoding (use a secure key in production)
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Dummy user credentials (replace with a database or secure storage in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "password"  # Replace with hashed passwords in production
    }
}

# Load the saved model
model_ref = bentoml.sklearn.get("admission_model:latest")
model_runner = model_ref.to_runner()

# Create a BentoML service
svc = bentoml.Service("admission_prediction_service", runners=[model_runner])

# Helper function to authenticate user
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or user["password"] != password:
        return False
    return user

# Helper function to create JWT tokens
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Helper function to verify JWT token
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Input schema for prediction
class AdmissionInput(BaseModel):
    GRE_Score: int  # Integer
    TOEFL_Score: int  # Integer
    University_Rating: int  # Integer
    SOP: float  # Float
    LOR: float  # Float
    CGPA: float  # Float
    Research: int  # Integer (binary: 0 or 1)

# Token endpoint for login
@svc.api(input=JSON(), output=JSON())
async def login(form_data: dict):
    username = form_data.get("username")
    password = form_data.get("password")
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Prediction endpoint
@svc.api(input=JSON(pydantic_model=AdmissionInput), output=JSON())
async def predict(input_data: AdmissionInput, context: bentoml.Context):
    # Extract token from headers
    authorization = context.request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.split(" ")[1]
    username = verify_token(token)

    # Convert input data to a NumPy array
    input_array = np.array([[input_data.GRE_Score, input_data.TOEFL_Score, input_data.University_Rating,
                             input_data.SOP, input_data.LOR, input_data.CGPA, input_data.Research]])
    
    # Make prediction
    prediction =  model_runner.async_run(input_array)
    
    # Return the prediction
    return {"username": username, "Chance of Admit": prediction[0]}