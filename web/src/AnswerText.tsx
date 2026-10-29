import type { ReactNode } from 'react'
import type { Citation } from './api'

// The answer is plain text with a little markdown (**bold**, "- " bullets)
// and citations written as «quote» [n]. Each citation is shown with whether
// the server found the quote in source n.

const CITATION = /«([^»]+)»\s*\[(\d+)\]|\*\*([^*]+)\*\*/g

function inline(line: string, citations: Citation[], onSource: (n: number) => void) {
  const parts: ReactNode[] = []
  let last = 0
  for (const m of line.matchAll(CITATION)) {
    parts.push(line.slice(last, m.index))
    if (m[3] !== undefined) {
      parts.push(<strong key={m.index}>{m[3]}</strong>)
    } else {
      const n = Number(m[2])
      const ok = citations.find((c) => c.quote === m[1] && c.source === n)?.valid
      parts.push(
        <span key={m.index} className={ok ? 'quote ok' : 'quote bad'}>
          «{m[1]}»
          <button className="ref" onClick={() => onSource(n)} title={ok ? 'הציטוט נמצא במקור' : 'הציטוט לא נמצא במקור'}>
            [{n}]{ok ? '✓' : '✗'}
          </button>
        </span>,
      )
    }
    last = m.index + m[0].length
  }
  parts.push(line.slice(last))
  return parts
}

export function AnswerText(props: { text: string; citations: Citation[]; onSource: (n: number) => void }) {
  const blocks: ReactNode[] = []
  let bullets: ReactNode[] = []
  const flush = () => {
    if (bullets.length) blocks.push(<ul key={`ul${blocks.length}`}>{bullets}</ul>)
    bullets = []
  }
  props.text.split('\n').forEach((raw, i) => {
    const line = raw.trim()
    if (!line) return flush()
    if (line.startsWith('- ')) {
      bullets.push(<li key={i}>{inline(line.slice(2), props.citations, props.onSource)}</li>)
    } else {
      flush()
      blocks.push(<p key={i}>{inline(line, props.citations, props.onSource)}</p>)
    }
  })
  flush()
  return <div className="answer-text">{blocks}</div>
}
