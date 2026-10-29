import { useState } from 'react'
import { AnswerText } from './AnswerText'
import type { Source } from './api'
import type { Step, Turn } from './chat'
import { Book, Check, ChevronDown, Clock, Copy, External, Refresh, Scale, Shield, X } from './icons'

// Section names that say nothing on their own; the chip shows the page instead.
const GENERIC = /^(שאלות נפוצות|שאלות ותשובות|בקצרה|כדאי לדעת|דוגמאות?|מידע נוסף)$/

function chipLabel(s: Source) {
  const leaf = s.section.split(' > ').pop() ?? s.title
  return GENERIC.test(leaf) || leaf === s.title ? s.title : leaf
}

function Steps({ steps, live }: { steps: Step[]; live: boolean }) {
  const [open, setOpen] = useState(false)
  if (!steps.length) return null
  const current = steps[steps.length - 1]
  if (live) {
    return (
      <div className="steps live">
        <span className="pulse" />
        <span className="shimmer">{current.label}</span>
        {current.detail && (
          <span className="step-detail" dir="auto">
            {current.detail}
          </span>
        )}
      </div>
    )
  }
  return (
    <div className="steps">
      <button className="steps-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>איך הגעתי לתשובה</span>
        <span className="muted">· {steps.length} שלבים</span>
        <ChevronDown size={15} />
      </button>
      {open && (
        <ol className="steps-list">
          {steps.map((s, i) => (
            <li key={i} className={`step step-${s.kind}`}>
              <span className="dot" />
              <div>
                <div>{s.label}</div>
                {s.detail && (
                  <div className="step-detail" dir="auto">
                    {s.detail}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function Sources({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState<number | null>(null)
  if (!turn.sources.length) return null
  const shown = turn.sources.find((s) => s.n === open)
  return (
    <div className="sources">
      <div className="source-chips">
        {turn.sources.map((s) => (
          <button
            key={`${s.n}-${s.section}`}
            className={`source-chip ${open === s.n ? 'active' : ''}`}
            onClick={() => setOpen(open === s.n ? null : s.n)}
            title={s.section}
          >
            <span className="chip-n">{s.n}</span>
            {/* Several chunks often come from one page, so show the section. */}
            <span className="chip-title">{chipLabel(s)}</span>
          </button>
        ))}
      </div>
      {shown && (
        <div className="source-card">
          <div className="source-card-head">
            <Book size={15} />
            <span>{shown.title}</span>
          </div>
          <div className="source-card-path">{shown.section.split(' > ').slice(1).join(' › ')}</div>
          <p>{shown.text}</p>
          <a href={shown.url} target="_blank" rel="noreferrer">
            לדף המלא בכל זכות
            {shown.updated ? ` · עודכן ${shown.updated}` : ''}
            <External size={13} />
          </a>
        </div>
      )}
    </div>
  )
}

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="msg user">
      <div className="bubble">{text}</div>
    </div>
  )
}

export function AssistantMessage({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  const [copied, setCopied] = useState(false)
  const live = turn.status === 'streaming'
  const r = turn.response
  const verified = r?.citations.filter((c) => c.valid).length ?? 0

  async function copy() {
    await navigator.clipboard.writeText(turn.draft)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="msg assistant">
      <div className="avatar">
        <Scale size={17} />
      </div>
      <div className="msg-body">
        <Steps steps={turn.steps} live={live} />
        <Sources turn={turn} />
        {r?.status === 'blocked' && <div className="note blocked">התשובה נחסמה כי הציטוטים בה לא אומתו</div>}
        {(turn.draft || (live && turn.steps.some((s) => s.kind === 'answer'))) && (
          <AnswerText
            text={turn.draft}
            streaming={live}
            citations={r ? r.citations : null}
            sources={turn.sources}
          />
        )}
        {turn.status === 'error' && <div className="note error">משהו השתבש: {turn.error}</div>}
        {turn.status === 'stopped' && <div className="note">העצירה בוצעה לפני שהתשובה הושלמה.</div>}

        {!live && (
          <div className="actions">
            {turn.draft && (
              <button className="icon-btn" onClick={copy} title="העתקה">
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            )}
            <button className="icon-btn" onClick={onRetry} title="לשאול שוב">
              <Refresh size={16} />
            </button>
            {r && (
              <div className="meta">
                {r.citations.length > 0 && (
                  <span className={verified === r.citations.length ? 'badge ok' : 'badge bad'}>
                    {verified === r.citations.length ? <Check size={13} /> : <X size={13} />}
                    {verified}/{r.citations.length} ציטוטים אומתו
                  </span>
                )}
                {r.review.decision && (
                  <span className="badge" title={r.review.decision.reason}>
                    <Shield size={13} />
                    מבקר
                  </span>
                )}
                <span className="badge plain">
                  <Clock size={13} />
                  {r.seconds} ש׳ · ${r.cost_usd.toFixed(4)}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
