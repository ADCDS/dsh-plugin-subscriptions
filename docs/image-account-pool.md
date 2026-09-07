# Image account scheduling

`image_generate` sends both generation and editing requests through `ImageAccountPool`. It reads the same live provider account list as chat, but maintains separate image health and affinity: chat catalogs and chat usage snapshots do not prove image entitlement or quota.

The first request tries accounts in default-first order. Successful requests establish per-session, per-provider affinity shared by generation and editing. Explicit HTTP 429, 401/403, or 402/404 rejection can move to another account of the selected provider. A 401 refreshes that account once before switching. Provider reset disclosures determine cooldowns when available; otherwise existing pool cooldown constants apply. Login/logout clears image health and affinity. Removed accounts cannot remain candidates.

Parameter errors, transport failures, timeouts, 5xx, and response decoding/storage failures do not automatically resend an image request. An upstream request may already have produced an image in those cases. Exhaustion never changes providers. The existing fallback to another provider when the preferred provider has no login remains unchanged.

`pool.enabled: false` or `pool.autoAccounts: false` disables automatic image account scheduling (`autoFamilies` is accepted as the legacy alias). Chat `strategy`, `families`, and `tiers` do not control image requests.

## Validation — 2026-09-07

- `pnpm build` passed; full tests: 413 passed, 6 skipped.
- Browser inputs in a fresh local DSH web session exercised real subscriptions, native tool calls, and inline image results.
- GPT generation succeeded. For the subsequent edit, observation of actual HTTP responses showed default account A returning **429**, then account B returning **200** on `/backend-api/codex/images/edits` within one model-requested tool call.
- A later new GPT generation sent no references and went directly to account B, returning **200** on `/backend-api/codex/images/generations`.
- Grok generation and editing both returned **200** on their respective endpoints. Editing selected the Grok source even with GPT images also in the conversation.
- Visual inspection confirmed red-to-green cube editing with the blue sphere retained, and red-to-blue cylinder editing. New generations did not inherit old references.
- Automated cases cover exhaustion, provider reset hints, sticky/removed accounts, auth-state clearing, bounded 401 recovery, disabled pooling, provider isolation, cancellation, and no retry after ambiguous failures.

The GPT endpoint returned dimensions different from the requested 1024×1024 during this run (1324×1188 and 1254×1254). The tool retained the returned image and its actual dimensions; account scheduling does not change size semantics. Browser checks used one reference per edit. Multi-reference and nested tool execution remain covered by automated tests.
