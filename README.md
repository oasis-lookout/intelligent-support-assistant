# Intelligent Assistant for Customer Support

A Retrieval-Augmented Generation (RAG) assistant for **financial-services support tickets**. One pipeline
(ingestion → retrieval with confidence guardrails → conversation memory → session security → per-business
isolation) that deploys two ways so they can be compared on throughput, latency, cost and operational burden:

| | **AWS-managed** (`DEPLOYMENT_MODE=aws`) | **Self-hosted** (`DEPLOYMENT_MODE=selfhosted`) | **Local / CI** (`local`) |
|---|---|---|---|
| Embeddings | Titan Text Embeddings V2 (Bedrock) | OpenAI-compatible server (e.g. vLLM, `bge-m3`) | hashed bag-of-words (offline) |
| Generation | Bedrock Converse API | vLLM (OpenAI-compatible) | extractive stub (offline) |
| Vector store | Qdrant on EKS | Qdrant on Kubernetes | in-process NumPy |
| State | RDS MySQL | MySQL | SQLite |
| Raw data | S3 | S3 / mounted volume | local CSV |

The code is written against small interfaces (`Embedder`, `LLM`, `VectorStore`), so the comparison is
apples-to-apples: only the providers change.

## Quick start (offline - no AWS account, no GPU, no network)

```bash
pip install -e ".[all]"        # or just `pip install numpy fastapi uvicorn` for the offline path
make test                      # 29 tests, ~1 s
make ingest                    # data/support_tickets_1000.csv -> 2,000 chunks
make chat                      # terminal chat as a CMP-101 customer
make eval                      # retrieval / isolation / leakage / threshold-sweep report in reports/
```

Run the API:

```bash
make keys                      # prints a customer key and an agent key for CMP-101 (shown once)
make run                       # http://localhost:8080  (OpenAPI docs at /docs)

# 1) exchange the tenant API key for a short-lived session token (tenant + role come from the key)
curl -s -X POST localhost:8080/v1/sessions -H "X-API-Key: $CUSTOMER_KEY"
# 2) chat
curl -s -X POST localhost:8080/v1/chat -H "Authorization: Bearer $TOKEN" \
     -H 'Content-Type: application/json' -d '{"message": "My account is blocked and I cannot make any payments"}'
```

Response:

```json
{"answer": "...", "confidence": 0.34, "handoff_recommended": true, "fallback_reason": null,
 "sources": [], "timings_ms": {"retrieve": 3.1, "generate": 0.2, "total": 4.0}}
```

`sources` is populated **only for the agent role** (customers never receive ticket IDs).

Full local stack with Qdrant + MySQL: `make up`, then
`docker compose run --rm tools sa-ingest --source data/support_tickets_1000.csv`.

## Deploying

**AWS-managed:** push the image to ECR; create an IRSA role with `deploy/iam/bedrock-s3-policy.json`; set
`BEDROCK_LLM_MODEL_ID` (no default on purpose - pick a model enabled in your account/region); create the
secret (see `deploy/k8s/base/secret.example.yaml`); then
`kubectl apply -k deploy/k8s/overlays/aws-managed` and run `deploy/k8s/jobs/ingest-job.yaml`.

**Self-hosted:** `kubectl apply -k deploy/k8s/overlays/self-hosted` (adds vLLM; needs a GPU node group), set
`LLM_MODEL`, run the ingest job.

## Design decisions (the parts worth discussing in the dissertation)

1. **Tenant isolation by construction.** `company_id` is the tenant. `VectorStore.search()` *requires* a tenant and
   a visibility set - there is no code path that queries across tenants. The tenant comes from the signed session
   token (derived from the API key), never from the request body. The service additionally asserts that no hit
   from another tenant is returned.
2. **Audience-aware retrieval.** Each ticket becomes `public` chunks (the conversation) and `internal` chunks
   (internal notes). Customers retrieve `public` only; agents retrieve both. Internal notes in the dataset contain
   fraud rule IDs, AML patterns and "do not disclose" instructions, so they must never reach a customer.
3. **Defence in depth against leakage.** (a) visibility filter at retrieval, (b) system prompt rules,
   (c) an output guard that blocks internal vocabulary (rule codes, structuring/mule/STR terms, citation markers)
   in customer responses, (d) optional Bedrock Guardrails, (e) a prompt-injection input filter.
4. **Confidence gating.** If no chunk reaches `MIN_CONFIDENCE`, the LLM is *not called*: the assistant says it
   doesn't know and recommends a human. This saves cost and prevents confident nonsense. The threshold is
   embedder-specific - calibrate it with the sweep printed by `make eval`.
5. **Sensitive intents always recommend a human** (`handoff_recommended`): blocks, fraud, disputes, legal. The
   assistant never claims to perform account actions (out of scope by design).
6. **PII redaction before indexing and before storing chat history** (emails, phones, PAN, card/long numbers);
   last-4 references and INR amounts are preserved because they carry meaning.
7. **Session security.** API key → HMAC-signed, expiring token bound to `(session, tenant, role)`; the session row
   is checked on every call (revocation); only key hashes are stored; sessions can be deleted, expired ones purged.
8. **Graceful degradation.** Provider errors return a safe fallback instead of a 500.
9. **Follow-ups without an extra LLM call.** Very short questions are prefixed with the previous user message for
   retrieval (keeps latency/cost low for the comparison; swap for LLM query rewriting if recall suffers).

## Evaluation & deployment comparison

* `sa-eval` (offline-capable): recall@k / MRR, isolation (cross-tenant and internal-chunk leaks must be 0),
  in-domain vs out-of-domain answer rates across thresholds, adversarial leakage prompts. Run it under each
  `DEPLOYMENT_MODE` and compare the JSON reports in `reports/`.
* `sa-loadtest`: concurrency sweep (p50/p95/p99, throughput, errors) against a running API. Run it against both
  deployments with identical questions for the latency/throughput comparison.
* Cost and operational burden: record instance hours, Bedrock token usage, GPU hours and your own ops effort.

## What has and hasn't been verified

Verified in an offline sandbox: ingestion, chunking/visibility split, redaction, tokens/API keys, guardrails,
conversation store (SQLite), in-memory vector store, the full service flow on the 1,000-ticket dataset
(29 unit/e2e tests pass; `sa-ingest` and `sa-eval` run).

**Not executed** (no network or those packages in the sandbox): the FastAPI layer, the Bedrock / Qdrant / MySQL /
OpenAI-compatible adapters, Docker and Kubernetes manifests. They are written against the documented APIs but
expect to fix small integration issues on first run - start with `make up`, then `/readyz`.

Caveats: the offline embedder is *lexical* and the extractive LLM only quotes past replies - they validate
plumbing and isolation, not answer quality. The dataset is synthetic and template-generated, so retrieval
metrics are optimistic; blend in real or paraphrased tickets before drawing quality conclusions.

## Production hardening checklist

Rate limiting / WAF in front of the API · Secrets Manager for `SESSION_SECRET`, DB credentials and API keys ·
TLS to RDS (`?ssl_ca=`) · pin image versions · Qdrant auth + backups · retention job calling
`purge_expired()` · PII-safe log review · move internal-notes access behind your IdP instead of static agent keys.

## Layout

```
src/support_assistant/
  ingestion/   loader (CSV/S3) · redact · chunker (public/internal) · pipeline
  providers/   base interfaces · local (offline) · bedrock · openai_compat (vLLM)
  stores/      memory (NumPy) · qdrant
  rag/         guardrails · prompt · service (online query path)
  memory/      conversations (SQLite / MySQL)
  security/    tokens (HMAC sessions) · apikeys
  api/         FastAPI app          cli/  ingest · keys · chat · evaluate · loadtest
deploy/        k8s (kustomize: base, aws-managed, self-hosted, ingest job) · iam policy
tests/         unittest, offline      docs/architecture.md
```
