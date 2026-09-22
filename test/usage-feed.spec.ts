/**
 * The loopback usage feed: configuration validation, token file handling,
 * request authentication, and assembling every account's usage so that one
 * failing account never fails the body.
 *
 * Every test binds an ephemeral port and writes its token under its own temp
 * directory, so nothing here touches `$DSH_HOME`.
 */

import { test, after } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  collectUsage,
  loadOrCreateToken,
  resolveUsageFeed,
  startUsageFeed,
  type UsageFeedSource,
} from '../src/usage-feed.js'

const dir = mkdtempSync(join(tmpdir(), 'usage-feed-'))
after(() => rmSync(dir, { recursive: true, force: true }))

const source: UsageFeedSource = {
  providers: ['claude', 'codex'],
  accounts: async provider => provider === 'claude'
    ? [{ key: 'a@example.com', account: 'a@example.com', plan: 'max', isDefault: true }]
    : [
        { key: 'one', account: 'one@example.com', isDefault: true },
        { key: 'two', account: 'two@example.com', isDefault: false },
      ],
  usage: async (provider, key) => {
    if (key === 'two') throw new Error('429 rate limited')
    return {
      supported: true,
      windows: [{ kind: 'session', usedPercent: provider === 'claude' ? 40 : 10, resetsAt: 1_000 }],
    }
  },
}

test('the feed is off unless enabled', () => {
  assert.equal(resolveUsageFeed(undefined, 'cfg'), undefined)
  assert.equal(resolveUsageFeed({ enabled: false }, 'cfg'), undefined)
  assert.deepEqual(resolveUsageFeed({ enabled: true }, 'cfg'), { host: '127.0.0.1', port: 8771 })
})

test('the feed refuses a non-loopback bind and an invalid port', () => {
  assert.throws(() => resolveUsageFeed({ enabled: true, host: '0.0.0.0' }, 'cfg'), /loopback/)
  assert.throws(() => resolveUsageFeed({ enabled: true, host: '192.168.1.5' }, 'cfg'), /loopback/)
  assert.throws(() => resolveUsageFeed({ enabled: true, port: 0 }, 'cfg'), /port/)
  assert.throws(() => resolveUsageFeed({ enabled: true, port: 70000 }, 'cfg'), /port/)
})

test('the token file is created 0600 and reused', async () => {
  const path = join(dir, 'nested', 'token')
  const first = await loadOrCreateToken(path)
  assert.ok(first.length >= 32)
  assert.equal(statSync(path).mode & 0o777, 0o600)
  assert.equal(await loadOrCreateToken(path), first)
})

test('a truncated token file is replaced and tightened to 0600', async () => {
  const path = join(dir, 'short-token')
  writeFileSync(path, 'short\n', { mode: 0o644 })
  const token = await loadOrCreateToken(path)
  assert.notEqual(token, 'short')
  assert.equal(readFileSync(path, 'utf8').trim(), token)
  assert.equal(statSync(path).mode & 0o777, 0o600)
})

test('one failing account becomes its own error; the rest still report', async () => {
  const body = await collectUsage(source, new AbortController().signal)
  assert.equal(body.providers.claude?.[0]?.usage?.windows?.[0]?.usedPercent, 40)
  const codex = body.providers.codex ?? []
  assert.equal(codex[0]?.usage?.windows?.[0]?.usedPercent, 10)
  assert.equal(codex[1]?.usage, undefined)
  assert.match(codex[1]?.error ?? '', /429/)
})

test('a provider whose account listing fails reports an error row', async () => {
  const body = await collectUsage({ ...source, accounts: async () => { throw new Error('store corrupt') } },
    new AbortController().signal)
  assert.match(body.providers.claude?.[0]?.error ?? '', /store corrupt/)
})

test('the served feed enforces the bearer token, method, path and browser origin', async () => {
  const tokenPath = join(dir, 'served-token')
  const warnings: string[] = []
  // Port 0 lets the kernel choose; read it back from the token-agnostic listing below.
  const probe = await import('node:net').then(net => new Promise<number>((resolve) => {
    const server = net.createServer()
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as { port: number }).port
      server.close(() => resolve(port))
    })
  }))
  const close = await startUsageFeed({ host: '127.0.0.1', port: probe }, source, w => warnings.push(w), tokenPath)
  try {
    const token = readFileSync(tokenPath, 'utf8').trim()
    const url = `http://127.0.0.1:${String(probe)}`

    assert.equal((await fetch(`${url}/usage`)).status, 401)
    assert.equal((await fetch(`${url}/usage`, { headers: { authorization: 'Bearer wrong' } })).status, 401)
    assert.equal((await fetch(`${url}/usage`, {
      headers: { authorization: `Bearer ${token}`, origin: 'https://evil.example' },
    })).status, 403)
    assert.equal((await fetch(`${url}/usage`, { method: 'POST', headers: { authorization: `Bearer ${token}` } })).status, 405)
    assert.equal((await fetch(`${url}/other`, { headers: { authorization: `Bearer ${token}` } })).status, 404)

    const ok = await fetch(`${url}/usage`, { headers: { authorization: `Bearer ${token}` } })
    assert.equal(ok.status, 200)
    const body = await ok.json() as { providers: Record<string, unknown[]> }
    assert.equal(body.providers.codex?.length, 2)
    assert.doesNotMatch(JSON.stringify(body), /accessToken|refreshToken/)
    assert.deepEqual(warnings, [])
  } finally {
    await close()
  }
})
