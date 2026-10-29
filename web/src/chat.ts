// Chat state: a turn is one question and the answer as it streams in.

import type { AskResponse, Source, StreamEvent } from './api'

export type Step = {
  kind: 'search' | 'sources' | 'answer' | 'retry' | 'blocked' | 'verified'
  label: string
  detail?: string
}

export type Turn = {
  id: string
  question: string
  status: 'streaming' | 'done' | 'error' | 'stopped'
  steps: Step[]
  draft: string // the answer text as it streams; replaced by the final one
  sources: Source[]
  response?: AskResponse
  error?: string
  startedAt: number
}

const newId = () => Math.random().toString(36).slice(2, 10)

export function newTurn(question: string): Turn {
  return {
    id: newId(),
    question,
    status: 'streaming',
    steps: [],
    draft: '',
    sources: [],
    startedAt: Date.now(),
  }
}

// One event in, the next turn state out.
export function applyEvent(t: Turn, e: StreamEvent): Turn {
  const step = (s: Step) => [...t.steps, s]
  switch (e.type) {
    case 'stage':
      if (e.stage === 'searching') return { ...t, steps: step({ kind: 'search', label: 'מחפש במקורות' }) }
      return { ...t, draft: '', steps: step({ kind: 'answer', label: 'כותב תשובה' }) }
    case 'sources':
      return {
        ...t,
        sources: e.sources,
        steps: step({ kind: 'sources', label: `נמצאו ${e.sources.length} קטעים רלוונטיים` }),
      }
    case 'token':
      return { ...t, draft: t.draft + e.text }
    case 'retry':
      return {
        ...t,
        draft: '',
        steps: step({ kind: 'retry', label: 'ציטוט לא נמצא במקור, כותב מחדש', detail: e.failed.join(' · ') }),
      }
    case 'blocked':
      return { ...t, steps: step({ kind: 'blocked', label: 'הציטוטים לא אומתו, התשובה נחסמה' }) }
    case 'done': {
      const r = e.response
      const ok = r.citations.filter((c) => c.valid).length
      return {
        ...t,
        status: 'done',
        response: r,
        draft: r.answer,
        sources: r.sources,
        steps: r.citations.length
          ? step({ kind: 'verified', label: `${ok} מתוך ${r.citations.length} ציטוטים אומתו מול המקור` })
          : t.steps,
      }
    }
    case 'error':
      return { ...t, status: 'error', error: e.message }
  }
}
