# Web client

A chat interface for the Rights RAG API. It streams each answer as it is
written, shows the pipeline's steps (search, answer, citation check, critic),
and marks every quote as verified or not once the server has checked it.

```bash
npm install
npm run dev        # http://localhost:5173, proxies /ask to the API on port 8000
```

Start the API first from the repository root:
`uv run uvicorn rights_rag.api:app`.

| File | What it does |
|---|---|
| `src/api.ts` | Types and the `POST /ask/stream` reader (server-sent events over `fetch`) |
| `src/chat.ts` | Turn state; turns stream events into steps, sources and answer text |
| `src/AnswerText.tsx` | Renders the answer: light markdown and `«quote» [n]` citations with hover cards |
| `src/Message.tsx` | A question and its answer: live steps, source chips, actions |
| `src/Composer.tsx` | The input box |
| `src/App.tsx` | Layout: top bar, empty state, thread |

Each question is answered on its own (the API has no conversation memory),
so the client keeps one conversation in memory and saves no history.
