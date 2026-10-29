import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Stop } from './icons'

type Props = {
  busy: boolean
  onSend: (text: string) => void
  onStop: () => void
}

export function Composer({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  // Grow with the text, up to a limit.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [text])

  useEffect(() => ref.current?.focus(), [])

  const ready = text.trim().length >= 3 && !busy
  function send() {
    if (!ready) return
    onSend(text.trim())
    setText('')
  }

  return (
    <div className="composer">
      <textarea
        ref={ref}
        value={text}
        rows={1}
        placeholder="שאלו על זכויות בעבודה…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            send()
          }
        }}
      />
      <div className="composer-bar">
        <span className="model">כל ציטוט נבדק מול המקור</span>
        {busy ? (
          <button className="send stop" onClick={onStop} title="עצירה" aria-label="עצירה">
            <Stop size={16} />
          </button>
        ) : (
          <button className="send" onClick={send} disabled={!ready} title="שליחה" aria-label="שליחה">
            <ArrowUp size={18} />
          </button>
        )}
      </div>
    </div>
  )
}
