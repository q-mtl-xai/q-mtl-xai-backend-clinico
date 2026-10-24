# Q-MTL-XAI Clinical Backend (v3.1)

Backend FastAPI con Supabase para el modo clínico.
Endpoints: /auth/*, /patients/*, /studies/analyze, /studies/explain

Variables de entorno:
- SUPABASE_URL
- SUPABASE_KEY
- SECRET_KEY
- GEMINI_API_KEY (opcional, para explicaciones LLM; si falta, usa fallback a reglas)
- GEMINI_MODEL (opcional, por defecto "gemini-2.5-flash")
