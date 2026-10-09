# Architecture

```mermaid
flowchart LR
  subgraph OFFLINE["Offline ingestion path (Kubernetes Job)"]
    S3[(S3 / CSV<br/>tickets)] --> L[Loader] --> R[PII redaction] --> C[Chunker<br/>public + internal]
    C --> E1[Embedder<br/>Titan V2 / bge-m3]
    E1 --> VDB[(Vector DB<br/>tenant_id + visibility payload)]
  end
  subgraph ONLINE["Online query path (FastAPI on EKS)"]
    U[Client] -->|X-API-Key| SES[POST /v1/sessions]
    SES -->|signed token| U
    U -->|Bearer token| CHAT[POST /v1/chat]
    CHAT --> IG[Input guard]
    IG --> E2[Embed query]
    E2 --> RET[Retrieve<br/>filter: tenant + visibility]
    RET --> VDB
    RET --> CG{Confidence<br/>gate}
    CG -- low --> FB[Fallback + hand-off]
    CG -- ok --> LLM[LLM<br/>Bedrock / vLLM]
    LLM --> OG[Output guard]
    OG --> U
    CHAT <--> DB[(RDS MySQL<br/>sessions + history)]
  end
```

## Request lifecycle

1. `POST /v1/sessions` - API key → `(tenant_id, role)`; a DB session row and an HMAC token are created.
2. `POST /v1/chat` - token verified; session row re-checked (expiry/revocation); identity becomes `SessionContext`.
3. Input guard (length, prompt-injection heuristics); message redacted before storage.
4. Retrieval query = question (or previous user message + question when very short).
5. Vector search filtered by `tenant_id` and role-based `visibility`; near-duplicate chunks removed.
6. Confidence gate: no chunk ≥ `MIN_CONFIDENCE` → fallback, **LLM not called**.
7. Prompt built from role-specific system prompt + `<context>` + bounded history; LLM generates.
8. Output guard (customer role) blocks internal vocabulary; provider errors → safe fallback.
9. Both turns persisted; response carries confidence, hand-off flag, timings (and sources for agents only).

## Data model

* `Chunk(chunk_id, tenant_id, ticket_id, visibility, text, ticket_status)`
* `visibility = public` (conversation) | `internal` (internal notes, prefixed with the customer's opening question)
* MySQL: `sessions(session_id, tenant_id, role, created_at, expires_at)`,
  `messages(session_id, seq, author, content, meta, created_at)`

## Threat model (summary)

| Threat | Mitigation |
|---|---|
| Cross-tenant data access | tenant in signed token; mandatory filter in store API; runtime assertion; tests + `sa-eval` |
| Customer sees internal notes / fraud rules | visibility filter, prompt rules, output guard, optional Bedrock Guardrails |
| Prompt injection via user text or retrieved text | input heuristics; context marked as data; no tool/action capabilities |
| Stolen / forged token | HMAC signature, short TTL, server-side session check, delete endpoint |
| PII at rest | redaction before indexing and before storing chat history |
| Hallucinated policy or promises | context-only prompt, confidence gating, no account actions |
