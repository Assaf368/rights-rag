// Types and calls for the API (see src/rights_rag/service.py and api.py).

export type Source = {
  n: number
  title: string
  section: string
  url: string
  updated: string | null
  text: string
}

export type Citation = { quote: string; source: number; valid: boolean }

export type AskResponse = {
  answer: string
  status: 'answered' | 'abstained' | 'blocked'
  disclaimer: string
  sources: Source[]
  citations: Citation[]
  cost_usd: number
  seconds: number
}

export type StreamEvent =
  | { type: 'stage'; stage: 'searching'; queries: string[] }
  | { type: 'stage'; stage: 'answering' }
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; text: string }
  | { type: 'retry'; failed: string[] }
  | { type: 'blocked' }
  | { type: 'done'; response: AskResponse }
  | { type: 'error'; message: string }

// POST /ask/stream answers with server-sent events. EventSource only does
// GET, so the body is read by hand: events are "data: {json}" blocks
// separated by a blank line.
export async function askStream(
  question: string,
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch('/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`השרת החזיר ${res.status}`)
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      const data = block
        .split('\n')
        .filter((l) => l.startsWith('data: '))
        .map((l) => l.slice(6))
        .join('\n')
      if (data) onEvent(JSON.parse(data) as StreamEvent)
    }
  }
}
