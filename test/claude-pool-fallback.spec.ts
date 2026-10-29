/**
 * The server-side fallback opt-in on the production path: a pool-routed Claude
 * model streamed through the real ClaudeAdapter. Parent turns and in-process
 * subagent turns both reach the adapter this way, so the request a pool member
 * sends must carry `fallbacks` and its beta. Fetch is stubbed; no network.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import './keep-alive.js'
import { MessageId } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { ClaudeAdapter, CLAUDE_SERVER_FALLBACK_BETA } from '../src/providers/claude.js'
import { AccountTokenManager } from '../src/providers/accounts.js'
import { PoolAdapter } from '../src/providers/pool.js'
import { poolKey } from '../src/providers/pool-family.js'
import type { PoolDefinition } from '../src/providers/pool-family.js'
import { PoolHealthRegistry } from '../src/providers/pool-health.js'
import { PoolUsageTracker } from '../src/providers/pool-usage.js'
import type { ClaudeSession } from '../src/auth/store.js'

const MODEL = 'claude-opus-5-5'
const session: ClaudeSession = { accessToken: 'at', refreshToken: 'rt', expiresAt: Date.now() + 3_600_000, scopes: 'scope' }

function tokens(): AccountTokenManager<ClaudeSession> {
  return new AccountTokenManager<ClaudeSession>({
    provider: 'claude',
    displayName: 'Test',
    makeOptions: () => ({ preemptMs: 0, refresh: s => Promise.resolve(s), isPermanent: () => false }),
    io: {
      list: () => Promise.resolve([{ key: 'acct', session }]),
      get: () => Promise.resolve(session),
      save: () => Promise.resolve(),
      remove: () => Promise.resolve(),
    },
  })
}

/** A minimal successful reply, so the pool commits to the member. */
const REPLY = [
  { type: 'message_start', message: { id: 'msg_1', model: MODEL, usage: { input_tokens: 3 } } },
  { type: 'content_block_start', index: 0, content_block: { type: 'text' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } },
  { type: 'content_block_stop', index: 0 },
  { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 1 } },
  { type: 'message_stop' },
].map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join('')

/** Stream one pool-routed turn; return the beta header and body the member sent. */
async function poolTurn(switchModelsOnFlag?: boolean): Promise<{ beta: string; body: Record<string, unknown>; text: string }> {
  const original = globalThis.fetch
  let beta = ''
  let body = ''
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    beta = new Headers(init?.headers).get('anthropic-beta') ?? ''
    body = String(init?.body)
    return new Response(REPLY, { status: 200, headers: { 'content-type': 'text/event-stream' } })
  }) as typeof globalThis.fetch
  try {
    const claude = new ClaudeAdapter({
      models: [{ id: MODEL, name: MODEL }],
      streamIdleTimeoutMs: 1000,
      tokens: tokens(),
      discovery: false,
      resolveCliVersion: async () => '2.1.999',
      ...switchModelsOnFlag === undefined ? {} : { switchModelsOnFlag },
    })
    const families = new Map<string, PoolDefinition>([
      [poolKey('claude', MODEL), { members: [{ provider: 'claude', account: 'acct', model: MODEL }] }],
    ])
    const pool = new PoolAdapter({
      adapters: { claude: claude as never },
      health: new PoolHealthRegistry(),
      usage: new PoolUsageTracker(() => undefined),
      strategy: 'priority',
      switchMargin: 2,
      defaultAccount: () => Promise.resolve('acct'),
      families: () => Promise.resolve(families),
      tiers: {},
      onWarn: () => {},
      retryDelay: async () => {},
    })
    const options: GenerateOptions = {
      provider: 'claude',
      model: MODEL,
      system: 'be brief',
      messages: [{ id: MessageId('m'), role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }],
      maxTokens: 1_000,
    }
    const chunks: StreamChunk[] = []
    for await (const chunk of pool.stream(options)) chunks.push(chunk)
    const text = chunks.filter(c => c.type === 'text-delta').map(c => (c as { text: string }).text).join('')
    return { beta, body: JSON.parse(body) as Record<string, unknown>, text }
  } finally {
    globalThis.fetch = original
  }
}

test('a pool-routed Claude turn carries the server-side fallback opt-in by default', async () => {
  const { beta, body, text } = await poolTurn()
  assert.equal(text, 'ok', 'the member reply streams through the pool')
  assert.equal(body.fallbacks, 'default')
  assert.ok(beta.split(',').includes(CLAUDE_SERVER_FALLBACK_BETA), 'the fallback beta rides with the Claude Code betas')
})

test('turning the option off leaves a pool-routed Claude turn without the opt-in', async () => {
  const { beta, body } = await poolTurn(false)
  assert.equal(body.fallbacks, undefined)
  assert.ok(!beta.split(',').includes(CLAUDE_SERVER_FALLBACK_BETA))
})
