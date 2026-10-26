
from pydantic import BaseModel
from typing import List, Union, Optional


class DoctorRegister(BaseModel):
    email: str
    password: str
    full_name: str
    medical_license: Optional[str] = None


class DoctorLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    doctor_id: str
    full_name: str


class PatientCreate(BaseModel):
    dni: str
    full_name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None


class PatientResponse(BaseModel):
    id: str
    dni: str
    full_name: str
    birth_date: Optional[str]
    gender: Optional[str]
    doctor_id: str


class StudyAnalyzeRequest(BaseModel):
    patient_id: str
    signal: Union[List[float], List[List[float]]]
    task: str = "isc"
    sampling_rate: int = 250


# Única definición de StudyResponse con los campos LLM integrados
class StudyResponse(BaseModel):
    study_id: str
    patient_id: str
    prediction: str
    confidence: float
    probabilities: dict
    sqi: str
    saliency: List[float]
    top_timesteps: List[int]
    latency_ms: float
    llm_explanation: Optional[str] = None
    llm_source: Optional[str] = None


class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    sqi: str
    top_timesteps: List[int]
    task: str = "isc"


class ExplainResponse(BaseModel):
    explicacion: str
    source: str


# === NUEVO: predicción sin guardar (para sandbox) ===
class PredictRequest(BaseModel):
    signal: Union[List[float], List[List[float]]]
    task: str = "isc"
    sampling_rate: int = 250


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    sqi: str
    saliency: List[float]
    top_timesteps: List[int]
    latency_ms: float
