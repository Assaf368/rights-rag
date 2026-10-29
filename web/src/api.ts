// Types and call for POST /ask (see src/rights_rag/service.py).

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
  review: {
    flag: string | null
    decision: { action: string; reason: string; flag: string } | null
    searched_again: boolean
    queries: string[]
  }
  cost_usd: number
  seconds: number
}

export async function ask(question: string, critic: boolean): Promise<AskResponse> {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, critic }),
  })
  if (!res.ok) throw new Error(`server returned ${res.status}`)
  return res.json()
}
