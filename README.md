# ChordGrid v0.1

MVP audio → grille d'accords synchronisée et éditable.

## Ce que fait cette V1

- Upload MP3/WAV/M4A/AAC/FLAC/OGG
- Détection de tempo (librosa)
- Estimation de tonalité (profils Krumhansl-Schmuckler simplifiés)
- Chromagramme CQT sur la composante harmonique
- Classification d'accord par beat avec alternatives + indice de confiance
- Grille 4/4, lecture synchronisée via WaveSurfer.js
- Correction manuelle d'un accord par double-clic
- Export TXT et JSON

> Limites V1 : mesure 4/4 supposée, segmentation par beat, pas encore de séparation de stems, slash chords ni analyse fonctionnelle.

## 1. Lancer le backend

Pré-requis : Python 3.11/3.12 et `ffmpeg` installé pour une compatibilité maximale avec MP3/M4A.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Tester : `http://localhost:8000/health`

## 2. Lancer le frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Ouvrir `http://localhost:5173`.

## 3. Déploiement recommandé

### Frontend — Netlify

Le fichier `netlify.toml` est déjà prêt. Dans Netlify, ajouter la variable :

```text
VITE_API_URL=https://URL-DE-TON-BACKEND
```

### Backend — Render / Railway / Fly.io

Le `backend/Dockerfile` est prêt. Déployer le dossier `backend` et définir :

```text
CORS_ORIGINS=https://TON-SITE.netlify.app
MAX_UPLOAD_MB=40
```

Commande de démarrage si la plateforme n'utilise pas Docker :

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Prochaines étapes musicales

1. Détection automatique 3/4, 6/8, changements de mesure.
2. Détection de basse séparée → slash chords (C/E, G/B…).
3. Séparation Harmonic/Percussive puis stems Demucs.
4. Vocabulaire jazz/gospel : 7b9, 7#9, 13, m11, maj9, sus13…
5. Lissage contextuel selon tonalité + accord précédent/suivant.
6. Sections intro/couplet/refrain/pont.
7. Degrés romains + Nashville Number System.
8. Bouton « Ouvrir dans MiScale » pour suggestions d'improvisation.
9. Export ChordPro, PDF, MusicXML/MIDI.

## Architecture

```text
frontend (React/Vite + WaveSurfer)
        ↓ multipart/form-data
backend (FastAPI)
        ↓
librosa: HPSS/harmonic → beat tracking → CQT chroma
        ↓
templates d'accords → confiance → JSON
        ↓
grille éditable dans le navigateur
```
