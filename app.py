
import os
import io
import time
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from models import MultiTaskEncoder
from preprocessing import preprocess_ecg
from sqi import predict_sqi_ecg
from xai_engine import compute_saliency_dynamic
from llm_explainer import explain_result
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_doctor,
)
from schemas import (
    DoctorRegister, DoctorLogin, TokenResponse,
    PatientCreate, PatientResponse,
    StudyAnalyzeRequest, StudyResponse,
    ExplainRequest, ExplainResponse,
    PredictRequest, PredictResponse,
    StudyUpdateRequest, StudyVoidRequest
)

# === Config ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DEVICE = torch.device("cpu")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Modelo ===
_BASE = os.path.dirname(__file__)
model = MultiTaskEncoder()
model.load_state_dict(torch.load(
    os.path.join(_BASE, "models", "c3_encoder_unificado_best.pt"),
    map_location=DEVICE,
))
model.eval()

CLASS_NAMES = {
    "latido": ["normal", "paced", "PVC", "SPB"],
    "ventana": ["normal", "paced", "PVC", "SPB"],
    "isc": ["normal", "ischemia"],
}

# === App ===
app = FastAPI(title="Q-MTL-XAI Clinical API", version="3.5.0") # Subimos versión
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_inference(signal, task, sampling_rate):
    if task not in ["latido", "ventana", "isc"]:
        raise HTTPException(400, f"Tarea inválida: {task}")

    sig = np.array(signal, dtype=np.float32)
    if task == "isc":
        if sig.ndim == 1 and len(sig) == 12 * 2500:
            sig = sig.reshape(12, 2500)
        elif sig.ndim == 1:
            sig = sig.reshape(1, -1)

    window_size = {"latido": 250, "ventana": 750, "isc": 2500}[task]
    processed = preprocess_ecg(sig, sampling_rate, 250, window_size=window_size)
    sqi_input = processed[0] if processed.ndim == 2 else processed
    sqi = predict_sqi_ecg(sqi_input)
    xai_result = compute_saliency_dynamic(model, processed, task=task, device=DEVICE)

    probs = xai_result["probabilities"]
    pred_idx = xai_result["pred_class"]
    prediction = CLASS_NAMES[task][pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "processed": processed,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[task][i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
        "sqi": sqi,
        "saliency": [round(float(v), 4) for v in xai_result["saliency"]],
        "top_timesteps": xai_result["top_timesteps"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
        "llm_available": bool(os.getenv("GEMINI_API_KEY", "")),
    }


# ---------- AUTH ----------
@app.post("/auth/register", response_model=TokenResponse)
def register(req: DoctorRegister):
    existing = supabase.table("doctors").select("*").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(400, "Email ya registrado")
    payload = {
        "email": req.email,
        "password_hash": hash_password(req.password),
        "full_name": req.full_name,
        "medical_license": req.medical_license,
    }
    res = supabase.table("doctors").insert(payload).execute()
    doctor = res.data[0]
    token = create_access_token({"sub": doctor["id"], "email": doctor["email"]})
    return TokenResponse(access_token=token, doctor_id=doctor["id"], full_name=doctor["full_name"])


@app.post("/auth/login", response_model=TokenResponse)
def login(req: DoctorLogin):
    res = supabase.table("doctors").select("*").eq("email", req.email).execute()
    if not res.data:
        raise HTTPException(401, "Credenciales inválidas")
    doctor = res.data[0]
    if not verify_password(req.password, doctor["password_hash"]):
        raise HTTPException(401, "Credenciales inválidas")
    token = create_access_token({"sub": doctor["id"], "email": doctor["email"]})
    return TokenResponse(access_token=token, doctor_id=doctor["id"], full_name=doctor["full_name"])


# ---------- PACIENTES ----------
@app.post("/patients", response_model=PatientResponse)
def create_patient(req: PatientCreate, doctor=Depends(get_current_doctor)):
    existing = supabase.table("patients").select("*").eq("dni", req.dni).execute()
    if existing.data:
        raise HTTPException(400, "DNI ya registrado")
    payload = {
        "doctor_id": doctor["sub"],
        "dni": req.dni,
        "full_name": req.full_name,
        "birth_date": req.birth_date,
        "gender": req.gender,
    }
    res = supabase.table("patients").insert(payload).execute()
    return PatientResponse(**res.data[0])


@app.get("/patients/by-dni/{dni}", response_model=PatientResponse)
def get_patient_by_dni(dni: str, doctor=Depends(get_current_doctor)):
    res = supabase.table("patients").select("*").eq("dni", dni).execute()
    if not res.data:
        raise HTTPException(404, "Paciente no encontrado")
    return PatientResponse(**res.data[0])


@app.get("/patients/{patient_id}/studies")
def list_studies(patient_id: str, doctor=Depends(get_current_doctor)):
    res = (supabase.table("studies")
           .select("*")
           .eq("patient_id", patient_id)
           .order("study_date", desc=True)
           .execute())
    return {"studies": res.data}


# ---------- ANÁLISIS CON PERSISTENCIA ----------
@app.post("/studies/analyze", response_model=StudyResponse)
def analyze_study(req: StudyAnalyzeRequest, doctor=Depends(get_current_doctor)):
    t0 = time.time()
    result = _run_inference(req.signal, req.task, req.sampling_rate)
    latency_ms = (time.time() - t0) * 1000

    study_id = str(uuid.uuid4())
    buf = io.BytesIO()
    np.save(buf, result["processed"])
    file_bytes = buf.getvalue()
    storage_path = f"{req.patient_id}/{study_id}.npy"
    try:
        supabase.storage.from_("ecg-files").upload(
            storage_path, file_bytes,
            {"content-type": "application/octet-stream", "upsert": "true"},
        )
    except Exception as e:
        print(f"[warn] Storage upload falló: {e}")

    record = {
        "id": study_id,
        "patient_id": req.patient_id,
        "doctor_id": doctor["sub"],
        "signal_file_path": storage_path,
        "ai_prediction": result["prediction"],
        "ai_confidence": result["confidence"],
        "sqi_score": result["sqi"],
        "medical_validation": req.medical_validation,
        "medical_notes": req.medical_notes,
        "saliency_data": result["saliency"],
        "top_timesteps": result["top_timesteps"],
        "task": req.task,
    }
    supabase.table("studies").insert(record).execute()

    return StudyResponse(
        study_id=study_id,
        patient_id=req.patient_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        sqi=result["sqi"],
        saliency=result["saliency"],
        top_timesteps=result["top_timesteps"],
        latency_ms=round(latency_ms, 2),
        llm_explanation=None,
        llm_source=None
    )


# ---------- PREDICCIÓN SIN PERSISTENCIA ----------
@app.post("/predict", response_model=PredictResponse)
def predict_only(req: PredictRequest, doctor=Depends(get_current_doctor)):
    t0 = time.time()
    result = _run_inference(req.signal, req.task, req.sampling_rate)
    latency_ms = (time.time() - t0) * 1000

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        sqi=result["sqi"],
        saliency=result["saliency"],
        top_timesteps=result["top_timesteps"],
        latency_ms=round(latency_ms, 2),
    )


# ---------- DETALLE DE UN ESTUDIO ----------
@app.get("/studies/{study_id}")
def get_study(study_id: str, doctor=Depends(get_current_doctor)):
    res = supabase.table("studies").select("*").eq("id", study_id).execute()
    if not res.data:
        raise HTTPException(404, "Estudio no encontrado")
    return res.data[0]


# ---------- NUEVO: DESCARGAR LA SEÑAL ORIGINAL DEL STORAGE ----------
@app.get("/studies/{study_id}/signal")
def get_study_signal(study_id: str, doctor=Depends(get_current_doctor)):
    res = supabase.table("studies").select("signal_file_path").eq("id", study_id).execute()
    if not res.data:
        raise HTTPException(404, "Estudio no encontrado")
    
    file_path = res.data[0]["signal_file_path"]
    
    try:
        file_bytes = supabase.storage.from_("ecg-files").download(file_path)
        buffer = io.BytesIO(file_bytes)
        signal = np.load(buffer, allow_pickle=True)
        return {"signal": signal.tolist()}
    except Exception as e:
        raise HTTPException(500, f"Error al descargar la señal: {str(e)}")


# ---------- EXPLICACIÓN ASISTIDA (Borrador) ----------
@app.post("/studies/explain", response_model=ExplainResponse)
def explain_study(req: ExplainRequest, doctor=Depends(get_current_doctor)):
    result = explain_result(
        prediction=req.prediction, confidence=req.confidence,
        sqi=req.sqi, top_timesteps=req.top_timesteps, task=req.task,
    )
    return ExplainResponse(explicacion=result["explicacion"], source=result["source"])


# ---------- EXPLICACIÓN POR ID Y GUARDAR EN BD ----------
@app.post("/studies/{study_id}/explain", response_model=ExplainResponse)
def explain_study_by_id(study_id: str, doctor=Depends(get_current_doctor)):
    res = supabase.table("studies").select("*").eq("id", study_id).execute()
    if not res.data:
        raise HTTPException(404, "Estudio no encontrado")

    study = res.data[0]

    if study.get("llm_explanation"):
        return ExplainResponse(
            explicacion=study["llm_explanation"],
            source=study.get("llm_source", "llm")
        )

    top_ts = study.get("top_timesteps") or []
    sqi_val = study.get("sqi_score") or "Fair"
    conf_val = study.get("ai_confidence") or 0.0
    task_val = study.get("task") or "isc"

    result = explain_result(
        prediction=study["ai_prediction"],
        confidence=conf_val,
        sqi=sqi_val,
        top_timesteps=top_ts,
        task=task_val
    )

    supabase.table("studies").update({
        "llm_explanation": result["explicacion"],
        "llm_source": result["source"]
    }).eq("id", study_id).execute()

    return ExplainResponse(explicacion=result["explicacion"], source=result["source"])


# ---------- EDICIÓN Y ANULACIÓN ----------
@app.patch("/studies/{study_id}")
def update_study_validation(study_id: str, req: StudyUpdateRequest, doctor=Depends(get_current_doctor)):
    res = supabase.table("studies").select("*").eq("id", study_id).execute()
    if not res.data:
        raise HTTPException(404, "Estudio no encontrado")

    supabase.table("studies").update({
        "medical_validation": req.medical_validation,
        "medical_notes": req.medical_notes
    }).eq("id", study_id).execute()

    return {"status": "ok", "message": "Validación y notas actualizadas"}


@app.patch("/studies/{study_id}/void")
def void_study(study_id: str, req: StudyVoidRequest, doctor=Depends(get_current_doctor)):
    res = supabase.table("studies").select("*").eq("id", study_id).execute()
    if not res.data:
        raise HTTPException(404, "Estudio no encontrado")

    supabase.table("studies").update({
        "medical_validation": "Anulado",
        "void_reason": req.void_reason
    }).eq("id", study_id).execute()

    return {"status": "ok", "message": "Estudio anulado correctamente"}
