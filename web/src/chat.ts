// Chat state: a conversation is a list of turns; a turn is one question and
// the answer as it streams in. Saved in localStorage, so it stays per browser.

import type { AskResponse, Decision, Source, StreamEvent } from './api'

export type Step = {
  kind: 'search' | 'sources' | 'answer' | 'retry' | 'blocked' | 'review' | 'critic' | 'search_again' | 'verified'
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
  decision?: Decision
  response?: AskResponse
  error?: string
  startedAt: number
}

export type Conversation = { id: string; title: string; turns: Turn[]; updatedAt: number }

const ACTION_LABEL: Record<string, string> = {
  accept: 'השאיר את התשובה',
  retry_with_rewrite: 'ביקש לנסח מחדש ולחפש שוב',
  abstain: 'החליט שאין תשובה במקורות',
}

const FLAG_LABEL: Record<string, string> = {
  abstained: 'המודל לא מצא תשובה',
  retried: 'נדרש ניסיון חוזר בציטוטים',
  blocked: 'הציטוטים לא אומתו',
}

export const newId = () => Math.random().toString(36).slice(2, 10)

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
export function newConversation(turn: Turn): Conversation {
  return { id: newId(), title: turn.question.slice(0, 60), turns: [turn], updatedAt: Date.now() }
}

export function applyEvent(t: Turn, e: StreamEvent): Turn {
  const step = (s: Step) => [...t.steps, s]
  switch (e.type) {
    case 'stage':
      if (e.stage === 'searching') return { ...t, steps: step({ kind: 'search', label: 'מחפש במקורות' }) }
      if (e.stage === 'searching_again')
        return {
          ...t,
          steps: step({ kind: 'search_again', label: 'מחפש שוב בניסוח אחר', detail: e.queries.slice(1).join(' · ') }),
        }
      if (e.stage === 'answering') return { ...t, draft: '', steps: step({ kind: 'answer', label: 'כותב תשובה' }) }
      return {
        ...t,
        steps: step({ kind: 'review', label: 'המבקר בודק את התשובה', detail: FLAG_LABEL[e.flag] ?? e.flag }),
      }
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
    case 'critic':
      return {
        ...t,
        decision: e.decision,
        steps: step({
          kind: 'critic',
          label: `המבקר ${ACTION_LABEL[e.decision.action] ?? e.decision.action}`,
          detail: e.decision.reason,
        }),
      }
    case 'replace':
      return { ...t, draft: e.text }
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

const KEY = 'rights-rag:conversations'

export function load(): Conversation[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]') as Conversation[]
    // A turn that was streaming when the page closed did not finish.
    return raw.map((c) => ({
      ...c,
      turns: c.turns.map((t) => (t.status === 'streaming' ? { ...t, status: 'stopped' as const } : t)),
    }))
  } catch {
    return []
  }
}

export function save(conversations: Conversation[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(conversations.slice(0, 50)))
  } catch {
    // storage full or blocked: the chat still works, it just is not saved
  }
}
