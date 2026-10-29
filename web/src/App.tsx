import { useCallback, useEffect, useRef, useState } from 'react'
import { askStream } from './api'
import { applyEvent, load, newConversation, newTurn, save, type Conversation, type Turn } from './chat'
import { Composer } from './Composer'
import { ArrowDown, Briefcase, Clock, Coffee, Help, Menu, Plus, Scale, Trash } from './icons'
import { AssistantMessage, UserMessage } from './Message'
import './App.css'

const SUGGESTIONS = [
  { icon: Briefcase, text: 'עבדתי שנה וחצי בשכר שעתי ופוטרתי. כמה זמן מראש היו צריכים להודיע לי?' },
  { icon: Coffee, text: 'אני עובד משרדי 9 שעות ביום. מגיעה לי הפסקת אוכל?' },
  { icon: Clock, text: 'נשארתי 11 שעות, והמנהל אמר שאצא מוקדם מחר ולכן אין שעות נוספות. זה חוקי?' },
  { icon: Help, text: 'מותר למעסיק לדרוש ממני בדיקת פוליגרף?' },
]

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(load)
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [critic, setCritic] = useState(true)
  const [sidebar, setSidebar] = useState(false)
  const [atBottom, setAtBottom] = useState(true)
  const abort = useRef<AbortController | null>(null)
  const thread = useRef<HTMLDivElement>(null)

  const current = conversations.find((c) => c.id === currentId) ?? null
  const busy = current?.turns.some((t) => t.status === 'streaming') ?? false

  useEffect(() => save(conversations), [conversations])

  const updateTurn = useCallback((convId: string, turnId: string, f: (t: Turn) => Turn) => {
    setConversations((cs) =>
      cs.map((c) =>
        c.id !== convId ? c : { ...c, updatedAt: Date.now(), turns: c.turns.map((t) => (t.id === turnId ? f(t) : t)) },
      ),
    )
  }, [])

  // Follow the answer while it streams, unless the reader scrolled up.
  useEffect(() => {
    const el = thread.current
    if (el && atBottom) el.scrollTop = el.scrollHeight
  }, [conversations, atBottom])

  async function run(convId: string, turn: Turn) {
    const controller = new AbortController()
    abort.current = controller
    try {
      await askStream(turn.question, turn.critic, (e) => updateTurn(convId, turn.id, (t) => applyEvent(t, e)), controller.signal)
      updateTurn(convId, turn.id, (t) =>
        t.status === 'streaming' ? { ...t, status: 'error', error: 'החיבור נסגר לפני סוף התשובה' } : t,
      )
    } catch (err) {
      const stopped = controller.signal.aborted
      updateTurn(convId, turn.id, (t) =>
        stopped
          ? { ...t, status: 'stopped' }
          : { ...t, status: 'error', error: err instanceof Error ? err.message : String(err) },
      )
    }
  }

  function send(text: string) {
    const turn = newTurn(text, critic)
    let convId = currentId
    if (!current) {
      const conv = newConversation(turn)
      convId = conv.id
      setConversations((cs) => [conv, ...cs])
      setCurrentId(convId)
    } else {
      setConversations((cs) =>
        cs.map((c) => (c.id === convId ? { ...c, updatedAt: Date.now(), turns: [...c.turns, turn] } : c)),
      )
    }
    setAtBottom(true)
    run(convId!, turn)
  }

  function retry(turn: Turn) {
    if (!current || busy) return
    const fresh = { ...newTurn(turn.question, critic), id: turn.id }
    updateTurn(current.id, turn.id, () => fresh)
    run(current.id, fresh)
  }

  function remove(id: string) {
    setConversations((cs) => cs.filter((c) => c.id !== id))
    if (id === currentId) setCurrentId(null)
  }

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)

  return (
    <div className={`app ${sidebar ? 'sidebar-open' : ''}`} dir="rtl">
      <aside className="sidebar">
        <div className="brand">
          <Scale size={20} />
          <span>זכויות עובדים</span>
        </div>
        <button
          className="new-chat"
          onClick={() => {
            setCurrentId(null)
            setSidebar(false)
          }}
        >
          <Plus size={17} />
          שיחה חדשה
        </button>
        <nav className="history">
          {sorted.length > 0 && <div className="history-label">שיחות קודמות</div>}
          {sorted.map((c) => (
            <div key={c.id} className={`history-item ${c.id === currentId ? 'active' : ''}`}>
              <button
                className="history-title"
                onClick={() => {
                  setCurrentId(c.id)
                  setSidebar(false)
                }}
              >
                {c.title}
              </button>
              <button className="history-delete" onClick={() => remove(c.id)} title="מחיקה" aria-label="מחיקה">
                <Trash size={15} />
              </button>
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          תשובות מתוך 50 דפים של כל זכות. כל ציטוט נבדק מול המקור לפני שהוא מוצג כמאומת.
        </div>
      </aside>
      {sidebar && <div className="scrim" onClick={() => setSidebar(false)} />}

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu" onClick={() => setSidebar(true)} aria-label="תפריט">
            <Menu size={20} />
          </button>
          <span className="topbar-title">{current?.title ?? 'שיחה חדשה'}</span>
          <button className="icon-btn" onClick={() => setCurrentId(null)} title="שיחה חדשה" aria-label="שיחה חדשה">
            <Plus size={19} />
          </button>
        </header>

        {!current ? (
          <div className="empty">
            <div className="empty-inner">
              <div className="empty-logo">
                <Scale size={30} />
              </div>
              <h1>מה תרצו לדעת על הזכויות שלכם?</h1>
              <p className="muted">פיטורים, הודעה מוקדמת, שעות נוספות, חופשה, הבראה, לידה ועוד.</p>
              <Composer busy={busy} critic={critic} onCritic={setCritic} onSend={send} onStop={() => abort.current?.abort()} />
              <div className="suggestions">
                {SUGGESTIONS.map(({ icon: Icon, text }) => (
                  <button key={text} className="suggestion" onClick={() => send(text)}>
                    <Icon size={18} />
                    <span>{text}</span>
                  </button>
                ))}
              </div>
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
                {current.turns.map((t) => (
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
              <Composer busy={busy} critic={critic} onCritic={setCritic} onSend={send} onStop={() => abort.current?.abort()} />
              <p className="dock-note">כל שאלה נענית בנפרד מתוך המקורות. המידע כללי ואינו ייעוץ משפטי.</p>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
