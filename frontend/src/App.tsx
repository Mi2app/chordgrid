import { useEffect, useMemo, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import type { Analysis, ChordEvent } from './types'

const API_URL = import.meta.env.VITE_API_URL || 'https://chordgrid-backend.onrender.com'
const COMMON_CHORDS = ['C','Cm','C7','Cmaj7','Cm7','D','Dm','D7','Dmaj7','Dm7','E','Em','E7','Emaj7','Em7','F','Fm','F7','Fmaj7','Fm7','F#','F#m','F#7','F#maj7','F#m7','G','Gm','G7','Gmaj7','Gm7','Ab','Abm','Ab7','Abmaj7','A','Am','A7','Amaj7','Am7','Bb','Bbm','Bb7','Bbmaj7','B','Bm','B7','Bmaj7','Bm7']

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

function confidenceClass(c: number) {
  if (c >= .78) return 'confidence high'
  if (c >= .58) return 'confidence medium'
  return 'confidence low'
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [stage, setStage] = useState('')
  const [mode, setMode] = useState<'simple' | 'standard' | 'jazz'>('standard')
  const [error, setError] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const waveformRef = useRef<HTMLDivElement>(null)
  const waveRef = useRef<WaveSurfer | null>(null)

  useEffect(() => {
    if (!waveformRef.current || !audioUrl) return
    waveRef.current?.destroy()
    const wave = WaveSurfer.create({
      container: waveformRef.current,
      url: audioUrl,
      height: 100,
      waveColor: '#4c5361',
      progressColor: '#f5c84c',
      cursorColor: '#ffffff',
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
    })
    wave.on('timeupdate', t => setCurrentTime(t))
    wave.on('play', () => setPlaying(true))
    wave.on('pause', () => setPlaying(false))
    waveRef.current = wave
    return () => wave.destroy()
  }, [audioUrl])

  const activeChord = useMemo(() => {
    return analysis?.chords.find(c => currentTime >= c.start && currentTime < c.end)
  }, [analysis, currentTime])

  const measures = useMemo(() => {
    if (!analysis) return [] as { number: number; beats: ChordEvent[] }[]
    const map = new Map<number, ChordEvent[]>()
    analysis.chords.forEach(ch => {
      const list = map.get(ch.measure) || []
      list.push(ch)
      map.set(ch.measure, list)
    })
    return [...map.entries()].map(([number, beats]) => ({ number, beats }))
  }, [analysis])

  function onFileChange(f: File | null) {
    if (!f) return
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setFile(f)
    setAudioUrl(URL.createObjectURL(f))
    setAnalysis(null)
    setError('')
    setCurrentTime(0)
  }

  async function analyze() {
    if (!file) return
    setLoading(true)
    setError('')
    setStage('Réveil du moteur…')
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 240000)
    try {
      await fetch(`${API_URL}/health`, { signal: controller.signal })
      setStage('Upload et analyse harmonique…')
      const fd = new FormData()
      fd.append('file', file)
      fd.append('mode', mode)
      const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: fd, signal: controller.signal })
      const body = await res.json()
      if (!res.ok) throw new Error(body.detail || 'Analyse impossible')
      setAnalysis(body)
      setStage('')
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') setError('Analyse trop longue (> 4 min). Essaie un extrait plus court ou le mode Simple.')
      else setError(e instanceof Error ? e.message : 'Erreur inconnue')
    } finally {
      window.clearTimeout(timeout)
      setLoading(false)
      setStage('')
    }
  }

  function seek(ch: ChordEvent) {
    waveRef.current?.setTime(ch.start)
    setCurrentTime(ch.start)
  }

  function updateChord(id: number, chord: string) {
    if (!analysis) return
    setAnalysis({ ...analysis, chords: analysis.chords.map(c => c.id === id ? { ...c, chord } : c) })
    setEditingId(null)
  }

  function exportJson() {
    if (!analysis) return
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${file?.name.replace(/\.[^.]+$/, '') || 'chordgrid'}-chords.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  function exportText() {
    if (!analysis) return
    const lines = measures.map(m => {
      const beats = Array.from({ length: analysis.beatsPerBar }, (_, i) => m.beats.find(b => b.beat === i + 1)?.chord || '—')
      return `| ${beats.join('  ')} |`
    })
    const text = [`${analysis.filename}`, `${analysis.key} • ${analysis.tempo} BPM • ${analysis.timeSignature}`, '', ...lines].join('\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${file?.name.replace(/\.[^.]+$/, '') || 'chordgrid'}-grid.txt`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="app-shell">
      <header>
        <div>
          <div className="eyebrow">AUDIO → HARMONIE</div>
          <h1>ChordGrid</h1>
          <p>Transforme un morceau en grille d'accords synchronisée, éditable et exportable.</p>
        </div>
        <div className="version">DSP v0.3</div>
      </header>

      <main>
        <section className="panel upload-panel">
          <label className="dropzone">
            <input type="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" onChange={e => onFileChange(e.target.files?.[0] || null)} />
            <span className="drop-icon">♪</span>
            <strong>{file ? file.name : 'Choisir un morceau'}</strong>
            <small>MP3, WAV, M4A, AAC, FLAC ou OGG • 40 Mo max</small>
          </label>
          <div className="analysis-controls">
            <label className="mode-select">
              <span>Mode</span>
              <select value={mode} onChange={e => setMode(e.target.value as 'simple' | 'standard' | 'jazz')} disabled={loading}>
                <option value="simple">Simple pop</option>
                <option value="standard">Standard</option>
                <option value="jazz">Jazz / Gospel</option>
              </select>
            </label>
            <button className="primary" disabled={!file || loading} onClick={analyze}>
              {loading ? (stage || 'Analyse harmonique…') : 'Analyser le morceau'}
            </button>
          </div>
          {error && <div className="error">{error}</div>}
        </section>

        {audioUrl && (
          <section className="panel player-panel">
            <div className="player-top">
              <button className="play" onClick={() => waveRef.current?.playPause()}>{playing ? 'Ⅱ' : '▶'}</button>
              <div className="now">
                <span className="now-label">ACCORD ACTUEL</span>
                <strong>{activeChord?.chord || '—'}</strong>
              </div>
              <div className="time">{formatTime(currentTime)} / {formatTime(waveRef.current?.getDuration() || 0)}</div>
            </div>
            <div ref={waveformRef} className="waveform" />
          </section>
        )}

        {analysis && (
          <>
            <section className="stats">
              <div className="stat"><span>Tempo</span><strong>{analysis.tempo}</strong><small>BPM</small></div>
              <div className="stat"><span>Tonalité</span><strong>{analysis.key.replace(' major','').replace(' minor','m')}</strong><small>{Math.round(analysis.keyConfidence * 100)}% confiance</small></div>
              <div className="stat"><span>Mesure</span><strong>{analysis.timeSignature}</strong><small>V1 supposée</small></div>
              <div className="stat"><span>Durée</span><strong>{formatTime(analysis.duration)}</strong><small>{analysis.processingSeconds ? `analyse ${analysis.processingSeconds}s • ` : ''}{analysis.filename}</small></div>
            </section>

            <section className="panel grid-panel">
              <div className="section-title-row">
                <div><div className="eyebrow">GRILLE</div><h2>Accords par mesure</h2></div>
                <div className="actions"><button onClick={exportText}>TXT</button><button onClick={exportJson}>JSON</button></div>
              </div>
              <p className="hint">Clique sur un accord pour écouter exactement à cet endroit. Double-clique pour le corriger.</p>
              <div className="measure-grid">
                {measures.map(m => (
                  <div className="measure" key={m.number}>
                    <span className="measure-number">{m.number}</span>
                    <div className="beats">
                      {Array.from({ length: analysis.beatsPerBar }, (_, i) => {
                        const ch = m.beats.find(b => b.beat === i + 1)
                        if (!ch) return <div className="beat empty" key={i}>—</div>
                        const active = activeChord?.id === ch.id
                        return (
                          <div key={ch.id} className={`beat ${active ? 'active' : ''}`} onClick={() => seek(ch)} onDoubleClick={() => setEditingId(ch.id)}>
                            {editingId === ch.id ? (
                              <div className="editor" onClick={e => e.stopPropagation()}>
                                <input autoFocus defaultValue={ch.chord} list="chord-list" onKeyDown={e => {
                                  if (e.key === 'Enter') updateChord(ch.id, e.currentTarget.value.trim() || ch.chord)
                                  if (e.key === 'Escape') setEditingId(null)
                                }} onBlur={e => updateChord(ch.id, e.currentTarget.value.trim() || ch.chord)} />
                              </div>
                            ) : <strong>{ch.chord}</strong>}
                            <span className={confidenceClass(ch.confidence)}>{Math.round(ch.confidence * 100)}%</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <datalist id="chord-list">{COMMON_CHORDS.map(c => <option key={c} value={c} />)}</datalist>
              <div className="legend"><span><i className="dot high"/> fiable</span><span><i className="dot medium"/> à vérifier</span><span><i className="dot low"/> incertain</span></div>
            </section>

            <section className="notice">{analysis.warning}</section>
          </>
        )}
      </main>
    </div>
  )
}
