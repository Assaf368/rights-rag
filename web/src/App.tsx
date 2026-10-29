import { useState } from 'react'
import { ask, type AskResponse } from './api'
import { AnswerText } from './AnswerText'
import './App.css'

const EXAMPLES = [
  'עבדתי שנה וחצי בשכר שעתי ופוטרתי. כמה זמן מראש היו צריכים להודיע לי?',
  'אני עובד משרדי. מגיעה לי הפסקת אוכל?',
  'עפו עליי מהעבודה בלי שום התראה. מה מגיע לי בתכלס?',
  'מותר למעסיק לדרוש ממני בדיקת פוליגרף?',
]

const STATUS: Record<AskResponse['status'], string> = {
  answered: '',
  abstained: 'אין תשובה במקורות',
  blocked: 'התשובה נחסמה: הציטוטים לא אומתו',
}

const ACTIONS: Record<string, string> = {
  accept: 'השאיר את התשובה',
  retry_with_rewrite: 'ניסח מחדש וחיפש שוב',
  abstain: 'החליף בהימנעות',
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [critic, setCritic] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AskResponse | null>(null)
  const [open, setOpen] = useState<number | null>(null)

  async function submit(q = question) {
    if (q.trim().length < 3) return
    setQuestion(q)
    setLoading(true)
    setError('')
    setResult(null)
    setOpen(null)
    try {
      setResult(await ask(q, critic))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const verified = result?.citations.filter((c) => c.valid).length ?? 0

  return (
    <main dir="rtl">
      <header>
        <h1>זכויות עובדים</h1>
        <p className="sub">תשובות מתוך 50 דפים של כל זכות, עם ציטוטים שנבדקים בקוד</p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="שאלו שאלה על זכויות בעבודה..."
          rows={3}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <div className="row">
          <label className="toggle">
            <input type="checkbox" checked={critic} onChange={(e) => setCritic(e.target.checked)} />
            מבקר (Sonnet) לתשובות חשודות
          </label>
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? 'מחפש...' : 'שאל'}
          </button>
        </div>
      </form>

      {!result && !loading && (
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button key={q} onClick={() => submit(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      {error && <p className="error">שגיאה: {error}. האם השרת רץ על פורט 8000?</p>}

      {result && (
        <section className="result">
          {STATUS[result.status] && <div className={`status ${result.status}`}>{STATUS[result.status]}</div>}
          <AnswerText text={result.answer} citations={result.citations} onSource={setOpen} />

          <div className="meta">
            {result.citations.length > 0 && (
              <span>
                ציטוטים שאומתו: {verified}/{result.citations.length}
              </span>
            )}
            {result.review.decision && (
              <span title={result.review.decision.reason}>
                מבקר: {ACTIONS[result.review.decision.action] ?? result.review.decision.action}
              </span>
            )}
            {result.review.searched_again && <span title={result.review.queries.slice(1).join(' | ')}>חיפוש נוסף</span>}
            <span>
              {result.seconds} שניות · ${result.cost_usd.toFixed(4)}
            </span>
          </div>

          <h2>מקורות</h2>
          <ol className="sources">
            {result.sources.map((s) => (
              <li key={s.n} className={open === s.n ? 'open' : ''}>
                <button className="source-head" onClick={() => setOpen(open === s.n ? null : s.n)}>
                  <span className="n">{s.n}</span>
                  <span>{s.section}</span>
                </button>
                {open === s.n && (
                  <div className="source-body">
                    <p>{s.text}</p>
                    <a href={s.url} target="_blank" rel="noreferrer">
                      לדף בכל זכות{s.updated ? ` (עודכן ${s.updated})` : ''}
                    </a>
                  </div>
                )}
              </li>
            ))}
          </ol>
          <p className="disclaimer">{result.disclaimer}</p>
        </section>
      )}
    </main>
  )
}
