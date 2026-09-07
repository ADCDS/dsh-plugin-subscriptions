import { test } from 'node:test'
import assert from 'node:assert/strict'
import { LlmError } from '@deepseek-ai/dsh-llm'
import { AccountTokenManager } from '../src/providers/accounts.js'
import { ImageAccountPool } from '../src/providers/image-pool.js'
import { codexRateLimitReset } from '../src/providers/codex.js'

function fixture() {
  const entries = ['first', 'second'].map(key => ({ key, session: { accessToken: key, refreshToken: key, expiresAt: Date.now() + 3600000 } }))
  let refreshes = 0
  const tokens = new AccountTokenManager({ provider: 'codex', displayName: 'Test',
    makeOptions: () => ({ preemptMs: 0, refresh: async session => { refreshes++; return { ...session, accessToken: session.accessToken + '-fresh' } }, isPermanent: () => false }),
    io: { list: async () => entries, get: async key => entries.find(e => e.key === (key ?? entries[0]?.key))?.session,
      save: async (key, session) => { const entry = entries.find(e => e.key === key); if (entry) entry.session = session },
      remove: async key => { const index = entries.findIndex(e => e.key === key); if (index >= 0) entries.splice(index, 1) } } })
  return { tokens, entries, refreshes: () => refreshes }
}
const signal = new AbortController().signal
const quota = () => new Response(JSON.stringify({ error: { type: 'usage_limit_reached', resets_in_seconds: 3600 } }), { status: 429 })

test('image pool: 429 switches accounts; generate/edit share affinity and cooldown; auth clears health', async () => {
  const { tokens } = fixture()
  const warnings: string[] = []
  const pool = new ImageAccountPool({ onWarn: message => warnings.push(message) })
  const attempts: string[] = []
  const owner = {}
  const options = { provider: 'codex' as const, tokens, owner, signal, rateLimitReset: codexRateLimitReset,
    send: async (session: { accessToken: string }) => { attempts.push(session.accessToken); return session.accessToken === 'first' ? quota() : new Response('ok') } }
  assert.equal(await (await pool.request(options)).text(), 'ok')
  assert.deepEqual(attempts, ['first', 'second'])
  assert.equal(warnings.length, 1)
  assert.ok(!warnings[0].includes('first'), 'logs do not expose account keys or tokens')
  await pool.request(options)
  await pool.request({ ...options, owner: {} })
  assert.deepEqual(attempts, ['first', 'second', 'second', 'second'])
  pool.clear('codex', 'first')
  await pool.request(options)
  assert.deepEqual(attempts.slice(-2), ['first', 'second'])
})

test('image pool: all accounts exhausted is bounded and later calls respect provider reset', async () => {
  const { tokens } = fixture()
  const pool = new ImageAccountPool()
  let attempts = 0
  const options = { provider: 'codex' as const, tokens, signal, rateLimitReset: codexRateLimitReset,
    send: async () => { attempts++; return quota() } }
  for (let i = 0; i < 2; i++) {
    await assert.rejects(() => pool.request(options), (e: unknown) => e instanceof LlmError && e.code === 'RATE_LIMIT' && (e.failure.providerRetryAfterMs ?? 0) > 3500000)
  }
  assert.equal(attempts, 2)
})

test('image pool: a 401 refreshes once, then switches if the account remains unauthorized', async () => {
  const { tokens, refreshes } = fixture()
  const attempts: string[] = []
  const pool = new ImageAccountPool()
  await pool.request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset,
    send: async session => { attempts.push(session.accessToken); return new Response('', { status: session.accessToken.startsWith('first') ? 401 : 200 }) } })
  assert.deepEqual(attempts, ['first', 'first-fresh', 'second'])
  assert.equal(refreshes(), 1)
})

test('image pool: recovered 401 uses refreshed credentials without another account', async () => {
  const { tokens, refreshes } = fixture()
  const attempts: string[] = []
  await new ImageAccountPool().request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset,
    send: async session => { attempts.push(session.accessToken); return new Response('', { status: session.accessToken === 'first' ? 401 : 200 }) } })
  assert.deepEqual(attempts, ['first', 'first-fresh'])
  assert.equal(refreshes(), 1)
})

for (const status of [400, 408, 500, 504]) {
  test(`image pool: HTTP ${status} does not risk duplicate image generation`, async () => {
    const { tokens } = fixture()
    let attempts = 0
    await assert.rejects(() => new ImageAccountPool().request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset,
      send: async () => { attempts++; return new Response('failure', { status }) } }))
    assert.equal(attempts, 1)
  })
}

test('image pool: transport failures and cancellation do not switch accounts', async () => {
  const { tokens } = fixture()
  let attempts = 0
  const options = { provider: 'codex' as const, tokens, signal, rateLimitReset: codexRateLimitReset,
    send: async () => { attempts++; throw new TypeError('connection lost') } }
  await assert.rejects(() => new ImageAccountPool().request(options), /connection lost/)
  await assert.rejects(() => new ImageAccountPool().request({ ...options, signal: AbortSignal.abort() }))
  assert.equal(attempts, 1)
})

test('image pool: disabled pooling uses only default; removed sticky accounts are not reused', async () => {
  const { tokens, entries } = fixture()
  let attempts = 0
  await assert.rejects(() => new ImageAccountPool({ enabled: false }).request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset,
    send: async () => { attempts++; return quota() } }))
  assert.equal(attempts, 1)
  const pool = new ImageAccountPool()
  const owner = {}
  const tried: string[] = []
  const options = { provider: 'codex' as const, tokens, owner, signal, rateLimitReset: codexRateLimitReset,
    send: async (session: { accessToken: string }) => { tried.push(session.accessToken); return new Response('ok') } }
  await pool.request(options)
  entries.shift()
  await pool.request(options)
  assert.deepEqual(tried, ['first', 'second'])
})

test('image pool: independent provider cooldowns; empty account list has login hint', async () => {
  const { tokens, entries } = fixture()
  const pool = new ImageAccountPool()
  await assert.rejects(() => pool.request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset, send: async () => quota() }))
  assert.equal((await pool.request({ provider: 'grok', tokens, signal, rateLimitReset: codexRateLimitReset, send: async () => new Response('ok') })).ok, true)
  entries.splice(0)
  await assert.rejects(() => pool.request({ provider: 'codex', tokens, signal, rateLimitReset: codexRateLimitReset, send: async () => { throw new Error('unreachable') } }), /not logged in/)
})
