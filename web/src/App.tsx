import { useCallback, useEffect, useRef, useState } from 'react'
import { askStream } from './api'
import { applyEvent, newTurn, type Turn } from './chat'
import { Composer } from './Composer'
import { ArrowDown, Baby, Briefcase, Door, Plus, Scale, Sun } from './icons'
import { AssistantMessage, UserMessage } from './Message'
import './App.css'

// Questions the sources answer well (checked against the golden set).
const SUGGESTIONS = [
  { icon: Briefcase, text: 'עבדתי שנה וחצי בשכר שעתי ופוטרתי. כמה זמן מראש היו צריכים להודיע לי?' },
  { icon: Door, text: 'העיפו אותי מהעבודה מיד, בלי שום הודעה מוקדמת. מה מגיע לי?' },
  { icon: Sun, text: 'כמה ימי הבראה מגיעים לי בשנה הראשונה במגזר הפרטי?' },
  { icon: Baby, text: 'אשתי ילדה. כמה ימים מותר לי לא להגיע לעבודה, ומה משלמים לי עליהם?' },
]

// One conversation, kept in memory only: the API answers each question on
// its own, so there is no history worth saving or switching between.
export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [atBottom, setAtBottom] = useState(true)
  const abort = useRef<AbortController | null>(null)
  const thread = useRef<HTMLDivElement>(null)

  const busy = turns.some((t) => t.status === 'streaming')

  const updateTurn = useCallback((turnId: string, f: (t: Turn) => Turn) => {
    setTurns((ts) => ts.map((t) => (t.id === turnId ? f(t) : t)))
  }, [])

  // Follow the answer while it streams, unless the reader scrolled up.
  useEffect(() => {
    const el = thread.current
    if (el && atBottom) el.scrollTop = el.scrollHeight
  }, [turns, atBottom])

  async function run(turn: Turn) {
    const controller = new AbortController()
    abort.current = controller
    try {
      await askStream(turn.question, (e) => updateTurn(turn.id, (t) => applyEvent(t, e)), controller.signal)
      updateTurn(turn.id, (t) =>
        t.status === 'streaming' ? { ...t, status: 'error', error: 'החיבור נסגר לפני סוף התשובה' } : t,
      )
    } catch (err) {
      const stopped = controller.signal.aborted
      updateTurn(turn.id, (t) =>
        stopped
          ? { ...t, status: 'stopped' }
          : { ...t, status: 'error', error: err instanceof Error ? err.message : String(err) },
      )
    }
  }

  function send(text: string) {
    const turn = newTurn(text)
    setTurns((ts) => [...ts, turn])
    setAtBottom(true)
    run(turn)
  }

  function retry(turn: Turn) {
    if (busy) return
    const fresh = { ...newTurn(turn.question), id: turn.id }
    updateTurn(turn.id, () => fresh)
    run(fresh)
  }

  function reset() {
    abort.current?.abort()
    setTurns([])
  }

  return (
    <div className="app" dir="rtl">
      <main className="main">
        <header className="topbar">
          <div className="brand">
            <Scale size={20} />
            <span>זכויות עובדים</span>
          </div>
          {turns.length > 0 && (
            <button className="new-chat" onClick={reset}>
              <Plus size={17} />
              שיחה חדשה
            </button>
          )}
        </header>

        {turns.length === 0 ? (
          <div className="empty">
            <div className="empty-inner">
              <div className="empty-logo">
                <Scale size={30} />
              </div>
              <h1>מה תרצו לדעת על הזכויות שלכם?</h1>
              <p className="muted">פיטורים, הודעה מוקדמת, שעות נוספות, חופשה, הבראה, לידה ועוד.</p>
              <Composer busy={busy} onSend={send} onStop={() => abort.current?.abort()} />
              <div className="suggestions">
                {SUGGESTIONS.map(({ icon: Icon, text }) => (
                  <button key={text} className="suggestion" onClick={() => send(text)}>
                    <Icon size={18} />
                    <span>{text}</span>
                  </button>
                ))}
              </div>
              <p className="empty-note">
                תשובות מתוך 50 דפים של כל זכות. כל ציטוט נבדק מול המקור לפני שהוא מוצג כמאומת.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div
              className="thread"
              ref={thread}
              onScroll={(e) => {
                const el = e.currentTarget
                setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
              }}
            >
              <div className="thread-inner">
                {turns.map((t) => (
                  <div key={t.id} className="turn">
                    <UserMessage text={t.question} />
                    <AssistantMessage turn={t} onRetry={() => retry(t)} />
                  </div>
                ))}
              </div>
            </div>
            {!atBottom && (
              <button className="to-bottom" onClick={() => setAtBottom(true)} aria-label="לסוף השיחה">
                <ArrowDown size={18} />
              </button>
            )}
            <div className="dock">
              <Composer busy={busy} onSend={send} onStop={() => abort.current?.abort()} />
              <p className="dock-note">כל שאלה נענית בנפרד מתוך המקורות. המידע כללי ואינו ייעוץ משפטי.</p>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
