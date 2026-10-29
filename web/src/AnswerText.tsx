import { useState, type ReactNode } from 'react'
import type { Citation, Source } from './api'

// The answer is plain text with a little markdown (**bold**, "- " and "1. "
// lists, "#" headings) and citations written as «quote» [n]. A citation is
// shown as a highlighted quote with a numbered chip; hovering the chip shows
// the source. While streaming, quotes are "checking"; once the server has
// checked them they turn verified or failed.

const INLINE = /«([^»]+)»\s*\[(\d+)\]|\*\*([^*]+)\*\*/g

type Ctx = {
  citations: Citation[] | null // null while streaming: not checked yet
  sources: Source[]
}

function CitationChip({ quote, n, ctx }: { quote: string; n: number; ctx: Ctx }) {
  const [open, setOpen] = useState(false)
  const source = ctx.sources.find((s) => s.n === n)
  const valid = ctx.citations?.find((c) => c.quote === quote && c.source === n)?.valid
  const state = ctx.citations === null ? 'checking' : valid ? 'ok' : 'bad'
  const note = { checking: 'בודק את הציטוט…', ok: 'הציטוט נמצא במקור, מילה במילה', bad: 'הציטוט לא נמצא במקור' }[state]
  return (
    <span className={`cite ${state}`}>
      <span className="cite-quote">{quote}</span>
      <span
        className="cite-chip"
        tabIndex={0}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        {n}
        {open && source && (
          <span className="popover" role="tooltip">
            <span className="popover-title">{source.title}</span>
            <span className="popover-section">{source.section}</span>
            <span className={`popover-state ${state}`}>{note}</span>
          </span>
        )}
      </span>
    </span>
  )
}

function inline(text: string, ctx: Ctx): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(INLINE)) {
    out.push(text.slice(last, m.index))
    out.push(
      m[3] !== undefined ? (
        <strong key={m.index}>{m[3]}</strong>
      ) : (
        <CitationChip key={m.index} quote={m[1]} n={Number(m[2])} ctx={ctx} />
      ),
    )
    last = m.index + m[0].length
  }
  out.push(text.slice(last))
  return out
}

export function AnswerText({ text, streaming, ...ctx }: Ctx & { text: string; streaming: boolean }) {
  const blocks: ReactNode[] = []
  let list: { ordered: boolean; items: ReactNode[] } | null = null
  const flush = () => {
    if (list) {
      const Tag = list.ordered ? 'ol' : 'ul'
      blocks.push(<Tag key={`l${blocks.length}`}>{list.items}</Tag>)
    }
    list = null
  }
  const lines = text.split('\n')
  lines.forEach((raw, i) => {
    const line = raw.trim()
    const caret = streaming && i === lines.length - 1 ? <span className="caret" /> : null
    if (!line) return flush()
    const bullet = line.match(/^[-*•]\s+(.*)$/)
    const numbered = line.match(/^\d+[.)]\s+(.*)$/)
    const heading = line.match(/^#{1,4}\s+(.*)$/)
    if (bullet || numbered) {
      const ordered = Boolean(numbered)
      if (!list || list.ordered !== ordered) {
        flush()
        list = { ordered, items: [] }
      }
      list.items.push(
        <li key={i}>
          {inline((bullet ?? numbered)![1], ctx)}
          {caret}
        </li>,
      )
      return
    }
    flush()
    blocks.push(
      heading ? (
        <h3 key={i}>
          {inline(heading[1], ctx)}
          {caret}
        </h3>
      ) : (
        <p key={i}>
          {inline(line, ctx)}
          {caret}
        </p>
      ),
    )
  })
  flush()
  if (streaming && !text.trim()) blocks.push(<span key="caret" className="caret" />)
  return <div className="answer-text">{blocks}</div>
}
