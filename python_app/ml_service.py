# ml_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import uvicorn

app = FastAPI(title="Project Prediction API")

# Train model once at startup
def train_model():
    data = {
        'team_size': [3, 5, 8, 4, 6, 7, 3, 9, 5, 8, 2, 10, 4, 7, 6],
        'complexity_score': [2, 7, 9, 3, 6, 8, 1, 10, 4, 7, 1, 9, 5, 8, 6],
        'on_time': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    X = df[['team_size', 'complexity_score']]
    y = df['on_time']

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_split=3)
    model.fit(X, y)
    return model

# Global model instance
ml_model = train_model()

# Request/Response models
class ProjectRequest(BaseModel):
    team_size: int
    complexity_score: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_project(request: ProjectRequest):
    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'team_size': [request.team_size],
        'complexity_score': [request.complexity_score]
    })

    # Get prediction and probability
    prediction = ml_model.predict(input_data)[0]
    probability = ml_model.predict_proba(input_data)[0].max()

    message = "Project likely ON TIME" if prediction == 1 else "Project likely DELAYED"

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 2),
        message=message
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)