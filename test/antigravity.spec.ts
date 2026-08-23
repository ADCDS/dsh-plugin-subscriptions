/** Antigravity OAuth, catalog/quota, request conversion, and stream tests. */

import { test } from 'node:test'
import { ToolCallId } from '../src/compat.js'
import { AccountTokenManager } from '../src/providers/accounts.js'
import assert from 'node:assert/strict'
import { MessageId } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { AntigravitySession } from '../src/auth/store.js'
import {
  AntigravityAdapter,
  ANTIGRAVITY_AUTHORIZE_URL,
  ANTIGRAVITY_TOKEN_URL,
  ANTIGRAVITY_USERINFO_URL,
  antigravityFlow,
  antigravityGenerateURL,
  exchangeAntigravityCode,
  fetchAntigravityModels,
  fetchAntigravityUsage,
  refreshAntigravity,
  requestAntigravityContent,
} from '../src/providers/antigravity.js'
import type { FetchFn } from '../src/providers/common.js'
import {
  AntigravityStreamTranslator,
  streamAntigravity,
  toAntigravityRequest,
} from '../src/translate/antigravity.js'
import type { TranslatableMessage } from '../src/translate/resolved.js'

const oauth = { clientId: 'test-client.apps.example.invalid', clientSecret: 'test-secret' }
const runtime = { baseURL: 'https://antigravity.example.invalid', onboard: false }
const session: AntigravitySession = {
  accessToken: 'access-token',
  refreshToken: 'refresh-token',
  expiresAt: Date.now() + 3_600_000,
  projectId: 'project-123',
  account: 'user@example.invalid',
  plan: 'AI Pro',
}

interface RecordedCall {
  url: string
  init?: RequestInit
}

/** URL-routed injected fetch with call recording. */
function routed(routes: Record<string, unknown | Response>, calls: RecordedCall[] = []): FetchFn {
  return async (input, init) => {
    const url = String(input)
    calls.push({ url, ...init === undefined ? {} : { init } })
    const value = routes[url]
    if (value instanceof Response) return value
    if (value === undefined) throw new Error(`unexpected fetch ${url}`)
    return new Response(JSON.stringify(value), { status: 200, headers: { 'content-type': 'application/json' } })
  }
}

function message(role: Message['role'], content: ContentBlock[], source?: Message['source']): Message {
  return {
    id: MessageId(`m-${Math.random().toString(36).slice(2)}`),
    role,
    content,
    source: source ?? (role === 'assistant'
      ? { kind: 'model', provider: 'antigravity', model: 'gemini-3-flash' }
      : { kind: 'user' }),
  }
}

function options(messages: Message[]): GenerateOptions {
  return {
    provider: 'antigravity',
    model: 'gemini-3-flash',
    messages,
    system: 'Be useful.',
    maxTokens: 2048,
    temperature: 0.2,
    tools: [{
      name: 'bash',
      description: 'run a command',
      parameters: { type: 'object', properties: { cmd: { type: 'string' } } },
    }],
  }
}

test('Antigravity OAuth URL uses Google PKCE, offline access, and distinct Antigravity scopes', () => {
  const flow = antigravityFlow(oauth)
  const url = new URL(flow.buildAuthorizeUrl({
    redirectUri: 'http://localhost:51121/oauth-callback',
    state: 'state-1',
    nonce: 'nonce-1',
    pkce: { verifier: 'verifier', challenge: 'challenge' },
  }))
  assert.equal(url.origin + url.pathname, ANTIGRAVITY_AUTHORIZE_URL)
  assert.equal(url.searchParams.get('client_id'), oauth.clientId)
  assert.equal(url.searchParams.get('access_type'), 'offline')
  assert.equal(url.searchParams.get('code_challenge_method'), 'S256')
  assert.match(url.searchParams.get('scope') ?? '', /experimentsandconfigs/)
  assert.doesNotMatch(url.searchParams.get('scope') ?? '', /gemini-cli/i)
})

test('exchangeAntigravityCode stores tokens, project, account, and plan without real credentials', async () => {
  const calls: RecordedCall[] = []
  const fetchFn = routed({
    [ANTIGRAVITY_TOKEN_URL]: {
      access_token: 'fresh-access', refresh_token: 'fresh-refresh', expires_in: 3600, scope: 'scope-a',
    },
    [`${runtime.baseURL}/v1internal:loadCodeAssist`]: {
      cloudaicompanionProject: 'project-live', paidTier: { name: 'Google AI Pro' },
    },
    [ANTIGRAVITY_USERINFO_URL]: { email: 'person@example.invalid' },
  }, calls)
  const result = await exchangeAntigravityCode(
    'code-1', 'verifier-1', 'http://localhost:51121/oauth-callback', oauth, runtime, fetchFn,
  )
  assert.equal(result.accessToken, 'fresh-access')
  assert.equal(result.refreshToken, 'fresh-refresh')
  assert.equal(result.projectId, 'project-live')
  assert.equal(result.account, 'person@example.invalid')
  assert.equal(result.plan, 'Google AI Pro')
  const tokenForm = new URLSearchParams(String(calls[0].init?.body))
  assert.equal(tokenForm.get('client_id'), oauth.clientId)
  assert.equal(tokenForm.get('client_secret'), oauth.clientSecret)
  assert.equal(tokenForm.get('code_verifier'), 'verifier-1')
})

test('refreshAntigravity preserves a rotating-token omission and account metadata', async () => {
  const calls: RecordedCall[] = []
  const result = await refreshAntigravity(session, oauth, routed({
    [ANTIGRAVITY_TOKEN_URL]: { access_token: 'renewed', expires_in: 1800 },
  }, calls))
  assert.equal(result.accessToken, 'renewed')
  assert.equal(result.refreshToken, session.refreshToken)
  assert.equal(result.projectId, session.projectId)
  assert.equal(result.account, session.account)
  const form = new URLSearchParams(String(calls[0].init?.body))
  assert.equal(form.get('grant_type'), 'refresh_token')
  assert.equal(form.get('refresh_token'), session.refreshToken)
})

test('model discovery and quota display map fetchAvailableModels data', async () => {
  const modelsURL = `${runtime.baseURL}/v1internal:fetchAvailableModels`
  const loadURL = `${runtime.baseURL}/v1internal:loadCodeAssist`
  const modelsPayload = {
    models: {
      'gemini-3-flash': {
        displayName: 'Gemini 3 Flash', inputTokenLimit: 500_000,
        quotaInfo: { remainingFraction: 0.7, resetTime: '2026-08-24T00:00:00Z' },
        weeklyQuotaInfo: { remainingFraction: 0.4, resetTime: '2026-08-30T00:00:00Z' },
      },
    },
  }
  const fetchFn = routed({
    [modelsURL]: modelsPayload,
    [loadURL]: { paidTier: { name: 'AI Ultra', availableCredits: [{ creditAmount: 42 }] } },
  })
  const models = await fetchAntigravityModels(session, runtime, fetchFn)
  assert.deepEqual(models, [{
    id: 'gemini-3-flash',
    name: 'Gemini 3 Flash',
    contextWindow: 500_000,
    inputModalities: ['text', 'image'],
  }])
  const usage = await fetchAntigravityUsage(session, runtime, fetchFn)
  assert.equal(usage.supported, true)
  assert.equal(usage.plan, 'AI Ultra · 42 credits')
  assert.deepEqual(usage.windows?.map(window => [window.kind, window.scope, Math.round(window.usedPercent)]), [
    ['other', 'gemini-3-flash', 30],
    ['weekly', 'gemini-3-flash', 60],
  ])
})

test('request conversion carries system, images, tools, tool results, and signed tool replay', () => {
  const assistantSource = {
    kind: 'model' as const,
    provider: 'antigravity',
    model: 'gemini-3-flash',
    replayState: {
      response: { kind: 'antigravity', version: 1 },
      blocks: [{}, { thoughtSignature: 'signed-thought' }],
    },
  }
  const messages: TranslatableMessage[] = [
    message('user', [{ type: 'text', text: 'inspect' }]),
    message('assistant', [
      { type: 'text', text: 'running' },
      { type: 'tool-call', id: ToolCallId('call-1'), name: 'bash', arguments: '{"cmd":"ls"}' },
    ], assistantSource),
    message('user', [{
      type: 'tool-result', toolCallId: ToolCallId('call-1'), content: [{ type: 'text', text: '{"ok":true}' }],
    }], { kind: 'tool', callId: ToolCallId('call-1') }),
    {
      role: 'user',
      content: [{ type: 'image', mediaType: 'image/png', dataBase64: 'aGVsbG8=' }],
    },
  ]
  const payload = toAntigravityRequest(options(messages as Message[]), messages, 'project-123')
  assert.equal(payload.project, 'project-123')
  assert.equal(payload.request.systemInstruction?.parts[0].text, 'Be useful.')
  assert.equal(payload.request.tools?.[0].functionDeclarations[0].name, 'bash')
  const modelParts = payload.request.contents.find(content => content.role === 'model')?.parts ?? []
  assert.equal(modelParts[1].functionCall?.name, 'bash')
  assert.equal(modelParts[1].thoughtSignature, 'signed-thought')
  const result = payload.request.contents.flatMap(content => content.parts).find(part => part.functionResponse)
  assert.deepEqual(result?.functionResponse?.response, { ok: true })
  const image = payload.request.contents.flatMap(content => content.parts).find(part => part.inlineData)
  assert.equal(image?.inlineData?.data, 'aGVsbG8=')
})

test('stream translator emits reasoning, text, tool call, usage, finish, and replay signature', () => {
  const translator = new AntigravityStreamTranslator()
  const chunks = translator.push({
    response: {
      candidates: [{
        content: { parts: [
          { thought: true, text: 'think', thoughtSignature: 'sig-1' },
          { text: 'answer' },
          { functionCall: { id: 'call-7', name: 'bash', args: { cmd: 'pwd' } }, thoughtSignature: 'sig-2' },
        ] },
        finishReason: 'STOP',
      }],
      usageMetadata: { promptTokenCount: 10, cachedContentTokenCount: 2, candidatesTokenCount: 4, thoughtsTokenCount: 1 },
    },
  })
  assert.deepEqual(chunks.map(chunk => chunk.type), [
    'block-start', 'reasoning-delta', 'block-start', 'text-delta',
    'block-start', 'tool-call-delta', 'block-end', 'block-end', 'block-end', 'usage', 'finish',
  ])
  const usage = chunks.find(chunk => chunk.type === 'usage')
  assert.deepEqual(usage?.usage, { inputTokens: 8, outputTokens: 4, cacheReadTokens: 2, reasoningTokens: 1 })
  const finish = chunks.find(chunk => chunk.type === 'finish')
  assert.equal(finish?.reason.kind, 'tool-calls')
  const replay = finish?.type === 'finish'
    ? finish.replayState as { blocks?: unknown[] } | undefined
    : undefined
  assert.deepEqual(replay?.blocks, [
    { thoughtSignature: 'sig-1' }, {}, { thoughtSignature: 'sig-2' },
  ])
})

function byteStream(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text))
      controller.close()
    },
  })
}

test('streamGenerateContent SSE and generateContent URL/forwarding are both supported', async () => {
  assert.equal(
    antigravityGenerateURL(runtime.baseURL, true),
    `${runtime.baseURL}/v1internal:streamGenerateContent?alt=sse`,
  )
  assert.equal(
    antigravityGenerateURL(runtime.baseURL, false),
    `${runtime.baseURL}/v1internal:generateContent`,
  )
  const streamed: StreamChunk[] = []
  const frame = { response: { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] } }
  for await (const chunk of streamAntigravity(byteStream(`data: ${JSON.stringify(frame)}\n\n`))) streamed.push(chunk)
  assert.deepEqual(streamed.map(chunk => chunk.type), ['block-start', 'text-delta', 'block-end', 'finish'])

  const calls: RecordedCall[] = []
  const payload = toAntigravityRequest(options([message('user', [{ type: 'text', text: 'hi' }])]), [
    message('user', [{ type: 'text', text: 'hi' }]),
  ], session.projectId)
  await requestAntigravityContent(session, payload, false, runtime, routed({
    [`${runtime.baseURL}/v1internal:generateContent`]: { response: {} },
  }, calls))
  assert.equal(calls[0].init?.method, 'POST')
  assert.equal(new Headers(calls[0].init?.headers).get('authorization'), 'Bearer access-token')
})

/** Isolated accounts: no disk, OAuth, or real subscription calls. */
function accountTokens(refresh: (value: AntigravitySession) => Promise<AntigravitySession> = async value => value) {
  const accounts = new Map<string, AntigravitySession>([
    ['alice', { ...session, account: 'alice', accessToken: 'alice', projectId: 'project-alice' }],
    ['bob', { ...session, account: 'bob', accessToken: 'bob', projectId: 'project-bob' }],
  ])
  const tokens = new AccountTokenManager<AntigravitySession>({
    provider: 'antigravity', displayName: 'Antigravity test',
    makeOptions: () => ({ preemptMs: 0, refresh, isPermanent: () => false }),
    io: {
      list: async () => [...accounts].map(([key, session]) => ({ key, session })),
      get: async key => accounts.get(key ?? 'alice'),
      save: async (key, value) => { accounts.set(key, value) },
      remove: async key => { accounts.delete(key) },
    },
  })
  return { tokens, accounts }
}

test('Antigravity catalogs stay account-scoped and invalidate after account changes', async () => {
  const { tokens, accounts } = accountTokens()
  const reads: string[] = []
  const adapter = new AntigravityAdapter({
    tokens, models: [], discovery: true, streamIdleTimeoutMs: 1000, runtime,
    fetchFn: async (_input, init) => {
      const token = new Headers(init?.headers).get('authorization')!.slice(7)
      reads.push(token)
      return Response.json({ models: { [token]: { displayName: token, inputTokenLimit: token === 'alice' ? 100 : 200 } } })
    },
  })
  assert.deepEqual((await adapter.listModels('antigravity')).map(model => model.id), ['alice', 'bob'])
  assert.equal((await adapter.resolveOwnModel('antigravity', 'bob', 'bob')).context?.contextWindow, 200)
  assert.deepEqual((await adapter.listOwnModels('antigravity', 'alice')).map(model => model.id), ['alice'])
  assert.deepEqual(reads.sort(), ['alice', 'bob'])
  accounts.set('bob', { ...accounts.get('bob')!, accessToken: 'bob-new' })
  adapter.clearAccountCatalog('bob')
  assert.deepEqual((await adapter.listModels('antigravity')).map(model => model.id), ['alice', 'bob-new'])
  accounts.delete('alice')
  adapter.clearAccountCatalog('alice')
  assert.deepEqual((await adapter.listModels('antigravity')).map(model => model.id), ['bob-new'])
})

test('Antigravity retries a 401 with the selected account and refreshed project', async () => {
  const refreshed: string[] = []
  const { tokens, accounts } = accountTokens(async value => {
    refreshed.push(value.account!)
    return { ...value, accessToken: 'bob-refreshed', projectId: 'project-refreshed' }
  })
  const calls: { token: string | null; project: string }[] = []
  const adapter = new AntigravityAdapter({
    tokens, models: [], discovery: false, streamIdleTimeoutMs: 1000, runtime,
    fetchFn: async (_input, init) => {
      calls.push({ token: new Headers(init?.headers).get('authorization'), project: JSON.parse(String(init?.body)).project })
      if (calls.length === 1) return new Response('', { status: 401 })
      return new Response(byteStream(`data: ${JSON.stringify({ response: { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] } })}\n\n`))
    },
  })
  const chunks: StreamChunk[] = []
  for await (const chunk of adapter.streamAccount(options([]), 'bob')) chunks.push(chunk)
  assert.deepEqual(refreshed, ['bob'])
  assert.deepEqual(calls, [
    { token: 'Bearer bob', project: 'project-bob' },
    { token: 'Bearer bob-refreshed', project: 'project-refreshed' },
  ])
  assert.equal(accounts.get('alice')?.accessToken, 'alice')
  assert.equal(chunks.at(-1)?.type, 'finish')
})

test('Antigravity discovery propagates cancellation instead of serving a fallback catalog', async () => {
  const { tokens } = accountTokens()
  const controller = new AbortController()
  controller.abort()
  const adapter = new AntigravityAdapter({
    tokens, models: [{ id: 'fallback' }], discovery: true, streamIdleTimeoutMs: 1000, runtime,
    fetchFn: async (_input, init) => { init?.signal?.throwIfAborted(); throw new Error('missing signal') },
  })
  await assert.rejects(adapter.listOwnModels('antigravity', 'alice', controller.signal), { name: 'AbortError' })
})

test('Antigravity preserves tool-result images after parallel tool responses', () => {
  const id = ToolCallId('image-call')
  const messages: TranslatableMessage[] = [
    { role: 'assistant', content: [{ type: 'tool-call', id, name: 'inspect', arguments: '{}' }] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: id, content: [
      { type: 'text', text: 'screenshot' }, { type: 'image', mediaType: 'image/png', dataBase64: 'aW1hZ2U=' },
    ] }] },
  ]
  const parts = toAntigravityRequest(options([]), messages, session.projectId).request.contents.flatMap(content => content.parts)
  const responseIndex = parts.findIndex(part => part.functionResponse)
  const imageIndex = parts.findIndex(part => part.inlineData)
  assert.ok(responseIndex >= 0 && imageIndex > responseIndex)
  assert.equal(parts[imageIndex].inlineData?.data, 'aW1hZ2U=')
})

test('Antigravity fails over between accounts through the shared pool before emitting content', async () => {
  const { tokens } = accountTokens()
  const { PoolAdapter } = await import('../src/providers/pool.js')
  const { PoolHealthRegistry } = await import('../src/providers/pool-health.js')
  const { PoolUsageTracker } = await import('../src/providers/pool-usage.js')
  const { poolKey } = await import('../src/providers/pool-family.js')
  const visited: string[] = []
  let pool: InstanceType<typeof PoolAdapter> | undefined
  const adapter = new AntigravityAdapter({
    tokens, models: [{ id: 'gemini-3-flash' }], discovery: false, streamIdleTimeoutMs: 1000, runtime,
    pool: () => pool,
    fetchFn: async (_input, init) => {
      const token = new Headers(init?.headers).get('authorization')!
      visited.push(token)
      if (token === 'Bearer alice') return new Response('quota limited', { status: 429, headers: { 'retry-after': '60' } })
      return new Response(byteStream(`data: ${JSON.stringify({ response: { candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] } })}\n\n`))
    },
  })
  pool = new PoolAdapter({
    adapters: { antigravity: adapter }, health: new PoolHealthRegistry(),
    usage: new PoolUsageTracker(() => undefined), strategy: 'priority', switchMargin: 2,
    defaultAccount: () => tokens.defaultAccount(), onWarn: () => {}, tiers: {},
    families: async () => new Map([[poolKey('antigravity', 'gemini-3-flash'), { members: [
      { provider: 'antigravity', model: 'gemini-3-flash', account: 'alice' },
      { provider: 'antigravity', model: 'gemini-3-flash', account: 'bob' },
    ] }]]),
  })
  assert.equal((await adapter.resolveModel('antigravity', 'gemini-3-flash')).id, 'gemini-3-flash')
  const chunks: StreamChunk[] = []
  for await (const chunk of adapter.stream(options([]))) chunks.push(chunk)
  assert.deepEqual(visited, ['Bearer alice', 'Bearer bob'])
  assert.equal(chunks.at(-1)?.type, 'finish')
})
