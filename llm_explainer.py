
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:
        print(f"[LLM] No se pudo inicializar Gemini: {e}")
        return None


def _build_rules_explanation(prediction, confidence, sqi, top_timesteps, task):
    pred = prediction.lower()
    fs = 250
    total_len = {"latido": 250, "ventana": 750, "isc": 2500}.get(task, 2500)
    top_s = [round(t / fs, 2) for t in top_timesteps[:3]]

    def loc(t):
        rel = t / total_len
        return "inicio" if rel < 0.33 else ("medio" if rel < 0.66 else "final")
    ubic = ", ".join(dict.fromkeys(loc(t) for t in top_timesteps[:3]))

    if pred in ("ischemia", "isquemia"):
        hallazgo, foco = "Posible isquemia miocárdica", "segmento ST y onda T"
        accion = ("Revisar derivaciones V2-V5 buscando elevación o depresión del ST. "
                  "Correlacionar con clínica (dolor torácico, disnea) y considerar troponinas.")
    elif pred == "normal":
        hallazgo, foco = "Sin hallazgos isquémicos relevantes", "trazado basal"
        accion = ("Correlacionar con síntomas. Si hay sospecha clínica alta, "
                  "repetir ECG en 6-12 h o solicitar prueba de esfuerzo.")
    elif pred == "pvc":
        hallazgo, foco = "Contracción ventricular prematura (PVC)", "complejo QRS ancho"
        accion = ("Cuantificar la carga de PVC/hora. Considerar Holter 24 h si son "
                  "frecuentes o sintomáticos.")
    elif pred == "spb":
        hallazgo, foco = "Latido supraventricular prematuro (SPB)", "onda P adelantada"
        accion = ("Suele ser benigno. Descartar desencadenantes (cafeína, estrés, "
                  "trastornos electrolíticos).")
    elif pred == "paced":
        hallazgo, foco = "Latido con marcapasos", "espícula previa al QRS"
        accion = ("Verificar captura y sensado del dispositivo. Correlacionar con la "
                  "programación vigente del marcapasos.")
    else:
        hallazgo, foco = f"Clase '{prediction}'", "patrón general"
        accion = "Requiere revisión por especialista en cardiología."

    if confidence >= 0.85:
        conf = f"alta ({confidence:.1%})"
    elif confidence >= 0.65:
        conf = f"moderada ({confidence:.1%})"
    else:
        conf = f"baja ({confidence:.1%}) — interpretar con cautela"

    calidad = {
        "Good": "Good — señal de buena calidad",
        "Fair": "Fair — señal aceptable, revisar posibles artefactos",
        "Poor": "Poor — señal de baja calidad, considerar repetir adquisición",
    }.get(sqi, sqi)

    return (
        f"- **Hallazgo:** {hallazgo}\n"
        f"- **Confianza:** {conf}\n"
        f"- **Segmento analizado:** {foco} (t≈{top_s[0]}s, {top_s[1]}s, {top_s[2]}s, zona {ubic})\n"
        f"- **Calidad de señal (SQI):** {calidad}\n"
        f"- **Sugerencia clínica:** {accion}\n\n"
        f"*Esta es una interpretación asistida generada automáticamente; "
        f"no constituye un diagnóstico clínico.*"
    )


def _build_llm_prompt(prediction, confidence, sqi, top_timesteps, task):
    fs = 250
    total_len = {"latido": 250, "ventana": 750, "isc": 2500}.get(task, 2500)
    top_s = [round(t / fs, 2) for t in top_timesteps[:5]]
    return f"""Sos un cardiólogo asistente que ayuda a interpretar análisis automáticos de ECG.

Datos del análisis:
- Tarea: {task} (duración total: {total_len/fs:.1f} s)
- Predicción del modelo: {prediction}
- Confianza: {confidence:.2%}
- Calidad de señal (SQI): {sqi}
- Timesteps con mayor peso XAI (segundos): {top_s}

FORMATO DE SALIDA OBLIGATORIO — devolvé EXACTAMENTE 4 bullets con este formato, sin párrafos introductorios ni texto corrido:

- **Hallazgo:** [una línea con la predicción y su interpretación clínica breve]
- **Confianza:** [alta/moderada/baja + porcentaje, con nota si es <65%]
- **Segmento analizado:** [qué parte del ECG sustenta la predicción según los timesteps: ST, onda T, QRS, onda P, etc.]
- **Sugerencia clínica:** [qué debe revisar o correlacionar el médico, 1-2 frases máximo]

Reglas estrictas:
1. NO escribas párrafos corridos. SOLO los 4 bullets.
2. NO inventes datos que no estén arriba.
3. NO des diagnóstico definitivo.
4. La última línea, separada por una línea en blanco, debe ser exactamente:
"*Esta es una interpretación asistida generada automáticamente; no constituye un diagnóstico clínico.*"
"""


def explain_result(prediction, confidence, sqi, top_timesteps, task="isc"):
    client = _get_client()
    if client is not None:
        try:
            prompt = _build_llm_prompt(prediction, confidence, sqi, top_timesteps, task)
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            texto = (resp.text or "").strip()
            if texto:
                return {"explicacion": texto, "source": "llm"}
        except Exception as e:
            print(f"[LLM] Fallo Gemini, usando reglas: {e}")

    return {
        "explicacion": _build_rules_explanation(
            prediction, confidence, sqi, top_timesteps, task
        ),
        "source": "rules",
    }
