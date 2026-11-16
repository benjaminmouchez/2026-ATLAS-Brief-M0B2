"""Point d'entree principal de l'API FastAPI - Starter Kit Apprenants.

Ce fichier est le coeur de votre mission pour le backend de la plateforme IA-TOS.
Vous devez completer les routes ci-dessous en vous appuyant sur :
- Le package metier : iatos_core (agents, scoring, sentiment, logger, config)
- Le contrat d'interface : backend.schemas (Pydantic v2)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from iatos_core import __version__
from iatos_core.logger import logger
from iatos_core.ml.scoring import CreditScoringPredictor
from iatos_core.ml.sentiment import SentimentAnalyzer
from iatos_core.ml.transcription import AudioTranscriber
from iatos_core.agent.graph import run_agent_message

from backend.schemas import (
    AudioTranscribeResponse,
    ChatRequest,
    ChatResponse,
    CreditPredictRequest,
    CreditPredictResponse,
    HealthResponse,
    SentimentPredictRequest,
    SentimentPredictResponse,
)

# Services partages (Singletons en memoire)
scoring_predictor: CreditScoringPredictor = None
sentiment_analyzer: SentimentAnalyzer = None
audio_transcriber: AudioTranscriber = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application FastAPI.
    
    TODO (Apprenants) :
    1. Instancier le CreditScoringPredictor() et l'assigner a la variable globale scoring_predictor.
    2. Instancier le SentimentAnalyzer() et l'assigner a la variable globale sentiment_analyzer.
    3. Logger le succes de l'initialisation avec loguru.
    """
    global scoring_predictor, sentiment_analyzer, audio_transcriber
    logger.info("Demarrage des services FastAPI IA-TOS...")
    
    scoring_predictor = CreditScoringPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    audio_transcriber = AudioTranscriber()
    logger.info("Modeles de scoring, de sentiment et Whisper charges en memoire.")

    yield
    logger.info("Arret des services FastAPI IA-TOS.")


# Instanciation de l'application FastAPI
app = FastAPI(
    title="Plateforme IA-TOS - API Multi-Services",
    description="API exposant l'agent conversationnel LangGraph, le scoring de credit, l'analyse de sentiment et la transcription vocale Whisper.",
    version=__version__,
    lifespan=lifespan,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Supervision"])
def health_check():
    """Endpoint de surveillance (Healthcheck) pour Uptime Kuma ou Docker.
    
    TODO (Apprenants) :
    Retourner une instance de HealthResponse avec :
    - status="ok"
    - version=__version__
    - service="iatos-api"
    """
    return HealthResponse(status="ok", version=__version__, service="iatos-api")


@app.post("/chat", response_model=ChatResponse, tags=["Agent Conversationnel"])
def chat_endpoint(payload: ChatRequest):
    """Endpoint conversationnel interrogeant l'agent LangGraph outille.
    
    TODO (Apprenants) :
    1. Convertir l'historique Pydantic (payload.history) en liste de dictionnaires [{'role': ..., 'content': ...}].
    2. Appeler run_agent_message(user_message=payload.message, history=history_dicts).
    3. Retourner une instance de ChatResponse avec :
       - success (bool)
       - response (str)
       - tool_called (bool)
       - tool_results (list[str])
    4. Gerer les exceptions potentielles avec un logger.error et un HTTPException(500).
    """
    try:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in payload.history]
        resultat = run_agent_message(user_message=payload.message, history=history_dicts)
        return ChatResponse(**resultat)
    except Exception as exc:
        logger.error("Erreur sur /chat : {err}", err=exc)
        raise HTTPException(status_code=500, detail="Erreur interne de l'agent conversationnel.")


@app.post("/predict/credit", response_model=CreditPredictResponse, tags=["Scoring Tabulaire"])
def predict_credit(payload: CreditPredictRequest):
    """Endpoint de prediction du score de credit (baseline Scikit-Learn M0).
    
    TODO (Apprenants) :
    1. Recuperer l'instance globale de scoring_predictor (ou l'instancier si None).
    2. Appeler scoring_predictor.predict(...) avec montant_demande, duree_mois et revenu_mensuel.
    3. Retourner une instance de CreditPredictResponse avec les resultats.
    4. Gerer les exceptions potentielles avec un HTTPException(500).
    """
    global scoring_predictor
    try:
        if scoring_predictor is None:
            scoring_predictor = CreditScoringPredictor()
        resultat = scoring_predictor.predict(
            montant_demande=payload.montant_demande,
            duree_mois=payload.duree_mois,
            revenu_mensuel=payload.revenu_mensuel,
        )
        return CreditPredictResponse(**resultat)
    except Exception as exc:
        logger.error("Erreur sur /predict/credit : {err}", err=exc)
        raise HTTPException(status_code=500, detail="Erreur interne du service de scoring.")


@app.post("/predict/sentiment", response_model=SentimentPredictResponse, tags=["Traitement NLP"])
def predict_sentiment(payload: SentimentPredictRequest):
    """Endpoint d'analyse de polarite de sentiment (baseline NLTK VADER M0).
    
    TODO (Apprenants) :
    1. Recuperer l'instance globale de sentiment_analyzer (ou l'instancier si None).
    2. Appeler sentiment_analyzer.analyze(payload.texte).
    3. Retourner une instance de SentimentPredictResponse avec les resultats.
    4. Gerer les exceptions potentielles avec un HTTPException(500).
    """
    global sentiment_analyzer
    try:
        if sentiment_analyzer is None:
            sentiment_analyzer = SentimentAnalyzer()
        resultat = sentiment_analyzer.analyze(payload.texte)
        return SentimentPredictResponse(**resultat)
    except Exception as exc:
        logger.error("Erreur sur /predict/sentiment : {err}", err=exc)
        raise HTTPException(status_code=500, detail="Erreur interne du service de sentiment.")


@app.post("/audio/transcribe", response_model=AudioTranscribeResponse, tags=["Traitement Audio"])
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint de transcription d'une note vocale WAV (Whisper Tiny)."""
    global audio_transcriber
    try:
        if audio_transcriber is None:
            audio_transcriber = AudioTranscriber()
        contenu = await file.read()
        resultat = audio_transcriber.transcribe(contenu)
        return AudioTranscribeResponse(filename=file.filename, **resultat)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Erreur sur /audio/transcribe : {err}", err=exc)
        raise HTTPException(status_code=500, detail="Erreur interne du service de transcription.")