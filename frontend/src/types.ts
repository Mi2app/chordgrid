export type Alternative = { chord: string; score: number }

export type ChordEvent = {
  id: number
  start: number
  end: number
  beat: number
  measure: number
  chord: string
  baseChord?: string
  bass?: string | null
  bassConfidence?: number
  confidence: number
  alternatives: Alternative[]
}

export type Analysis = {
  filename: string
  duration: number
  tempo: number
  key: string
  keyConfidence: number
  timeSignature: string
  beatsPerBar: number
  chords: ChordEvent[]
  sections: unknown[]
  engine: string
  mode?: string
  processingSeconds?: number
  warning: string
}
