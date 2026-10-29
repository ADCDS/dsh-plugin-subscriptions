/**
 * Codex CLI credential import: reading `$CODEX_HOME/auth.json`, the synced
 * refresh that keeps the CLI and this plugin on one rotation of a shared login,
 * and the `login('codex', 'cli')` path. Every test runs against a private
 * `$CODEX_HOME` (and, where it reaches the session store, a private
 * `$DSH_HOME`), so nothing touches a developer's real Codex or harness login.
 */

import { test, after } from 'node:test'
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SubscriptionsAuthController } from '../src/index.js'
import { OAuthFlowManager } from '../src/auth/oauth-flow.js'
import { DeviceFlowManager } from '../src/auth/device-flow.js'
import { readClaudeCodeCredentials, refreshClaudeSynced } from '../src/auth/claude-code-creds.js'
import { readCodexCliCredentials, refreshCodexSynced } from '../src/auth/codex-cli-creds.js'
import { CODEX_CLIENT_ID } from '../src/providers/codex.js'
import { authFilePath, listAccounts } from '../src/auth/store.js'
import type { ClaudeSession, CodexSession } from '../src/auth/store.js'

const TEMP_DIRS: string[] = []
const PREEMPT_MS = 5 * 60_000

function tempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix))
  TEMP_DIRS.push(dir)
  return dir
}

after(() => {
  for (const dir of TEMP_DIRS) rmSync(dir, { recursive: true, force: true })
})

async function withEnv<T>(name: string, value: string, run: () => Promise<T>): Promise<T> {
  const saved = process.env[name]
  process.env[name] = value
  try {
    return await run()
  } finally {
    if (saved === undefined) delete process.env[name]
    else process.env[name] = saved
  }
}

/** Build an unsigned JWT (header.payload.sig, base64url). */
function jwt(payload: Record<string, unknown>): string {
  return `e30.${Buffer.from(JSON.stringify(payload)).toString('base64url')}.sig`
}

function idToken(accountId: string, email = 'dev@example.com'): string {
  return jwt({
    email,
    'https://api.openai.com/auth': { chatgpt_account_id: accountId, chatgpt_plan_type: 'pro', chatgpt_user_id: 'user-1' },
  })
}

/** An access token expiring `inMs` from now, tagged so tests can tell rotations apart. */
function accessToken(tag: string, inMs: number): string {
  return jwt({ exp: Math.trunc((Date.now() + inMs) / 1000), tag })
}

interface FileTokens { access: string; refresh: string; accountId?: string; id?: string }

/** A Codex `auth.json` as the CLI writes it, including fields this module must preserve. */
function authDoc(tokens: FileTokens, extra: Record<string, unknown> = {}): Record<string, unknown> {
  const accountId = tokens.accountId ?? 'acct-1'
  return {
    auth_mode: 'chatgpt',
    OPENAI_API_KEY: null,
    tokens: {
      id_token: tokens.id ?? idToken(accountId),
      access_token: tokens.access,
      refresh_token: tokens.refresh,
      oauth_provenance: { issuer: 'https://auth.openai.com', client_id: CODEX_CLIENT_ID },
      login_owner_id: 'owner-1',
      account_id: accountId,
    },
    last_refresh: '2026-10-01T00:00:00.000000000Z',
    ...extra,
  }
}

/** A private Codex home holding one `auth.json`; returns its directory. */
function codexHome(doc?: Record<string, unknown> | string): string {
  const dir = tempDir('codex-home-')
  if (doc !== undefined) {
    writeFileSync(join(dir, 'auth.json'), typeof doc === 'string' ? doc : JSON.stringify(doc, null, 2), { mode: 0o600 })
  }
  return dir
}

function readDoc(dir: string): Record<string, any> {
  return JSON.parse(readFileSync(join(dir, 'auth.json'), 'utf8')) as Record<string, any>
}

/** The session the plugin holds for an imported login with these tokens. */
function session(tokens: FileTokens, expiresInMs: number): CodexSession {
  return {
    accessToken: tokens.access,
    refreshToken: tokens.refresh,
    expiresAt: Date.now() + expiresInMs,
    accountId: tokens.accountId ?? 'acct-1',
    idToken: tokens.id ?? idToken(tokens.accountId ?? 'acct-1'),
    cliBound: true,
  }
}

/** A refresh grant that rotates to `tag` and records the session it spent. */
function rotatingRefresh(tag: string, spent: CodexSession[]) {
  return async (from: CodexSession): Promise<CodexSession> => {
    spent.push(from)
    return {
      accessToken: accessToken(tag, 10 * 24 * 3600_000),
      refreshToken: `rt-${tag}`,
      expiresAt: Date.now() + 10 * 24 * 3600_000,
      accountId: from.accountId,
      idToken: idToken(from.accountId),
    }
  }
}

// ---------------------------------------------------------------------------
// Reading the CLI's login
// ---------------------------------------------------------------------------

test('readCodexCliCredentials returns the ChatGPT login, marked CLI-bound', async () => {
  const access = accessToken('a1', 3600_000)
  const dir = codexHome(authDoc({ access, refresh: 'rt-1' }))
  await withEnv('CODEX_HOME', dir, async () => {
    const read = readCodexCliCredentials()
    assert.ok(read)
    assert.equal(read.accessToken, access)
    assert.equal(read.refreshToken, 'rt-1')
    assert.equal(read.accountId, 'acct-1')
    assert.equal(read.emailAddress, 'dev@example.com')
    assert.equal(read.planType, 'pro')
    assert.equal(read.cliBound, true)
    // Expiry comes from the access token's exp claim, to the second.
    assert.ok(Math.abs(read.expiresAt - (Date.now() + 3600_000)) < 2000)
  })
})

test('readCodexCliCredentials takes the account from the id token when account_id is absent', async () => {
  const doc = authDoc({ access: accessToken('a1', 3600_000), refresh: 'rt-1', accountId: 'acct-claim' })
  delete (doc.tokens as Record<string, unknown>).account_id
  await withEnv('CODEX_HOME', codexHome(doc), async () => {
    assert.equal(readCodexCliCredentials()?.accountId, 'acct-claim')
  })
})

test('readCodexCliCredentials rejects files that hold no usable ChatGPT login', async () => {
  const good = (): Record<string, unknown> => authDoc({ access: accessToken('a1', 3600_000), refresh: 'rt-1' })
  const cases: [string, Record<string, unknown> | string | undefined][] = [
    ['no auth.json', undefined],
    ['malformed JSON', '{not json'],
    ['an API-key login', { ...good(), auth_mode: 'apikey', tokens: null, OPENAI_API_KEY: 'sk-test' }],
    ['no tokens', { ...good(), tokens: null }],
    ['an empty refresh token', authDoc({ access: accessToken('a1', 3600_000), refresh: '' })],
    ['an access token without exp', authDoc({ access: jwt({ tag: 'no-exp' }), refresh: 'rt-1' })],
  ]
  const foreign = good()
  ;((foreign.tokens as Record<string, unknown>).oauth_provenance as Record<string, unknown>).client_id = 'app_someone_else'
  cases.push(['a login issued to another OAuth client', foreign])
  for (const [label, doc] of cases) {
    await withEnv('CODEX_HOME', codexHome(doc), async () => {
      assert.equal(readCodexCliCredentials(), undefined, label)
    })
  }
})

// ---------------------------------------------------------------------------
// The synced refresh
// ---------------------------------------------------------------------------

test('synced refresh writes the rotation back, preserving the CLI\'s other fields', async () => {
  const tokens = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
  const dir = codexHome(authDoc(tokens))
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    const next = await refreshCodexSynced(session(tokens, 60_000), rotatingRefresh('a2', spent), PREEMPT_MS)
    assert.equal(spent.length, 1)
    assert.equal(spent[0]?.refreshToken, 'rt-1')
    assert.equal(next.refreshToken, 'rt-a2')
    assert.equal(next.cliBound, true, 'the binding survives the refresh')
    const doc = readDoc(dir)
    assert.equal(doc.tokens.access_token, next.accessToken)
    assert.equal(doc.tokens.refresh_token, 'rt-a2')
    assert.equal(doc.tokens.id_token, next.idToken)
    assert.equal(doc.tokens.account_id, 'acct-1')
    assert.equal(doc.tokens.login_owner_id, 'owner-1')
    assert.deepEqual(doc.tokens.oauth_provenance, { issuer: 'https://auth.openai.com', client_id: CODEX_CLIENT_ID })
    assert.equal(doc.OPENAI_API_KEY, null)
    assert.equal(doc.auth_mode, 'chatgpt')
    assert.ok(Date.now() - Date.parse(doc.last_refresh) < 60_000, 'last_refresh records the rotation')
    if (process.platform !== 'win32') assert.equal(statSync(join(dir, 'auth.json')).mode & 0o777, 0o600)
  })
})

test('synced refresh adopts a rotation the CLI already made instead of spending a refresh token', async () => {
  const ours = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
  const cli = { access: accessToken('a2', 10 * 24 * 3600_000), refresh: 'rt-2' }
  const dir = codexHome(authDoc(cli))
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    const next = await refreshCodexSynced(session(ours, 60_000), rotatingRefresh('never', spent), PREEMPT_MS)
    assert.equal(spent.length, 0, 'no refresh token was spent')
    assert.equal(next.accessToken, cli.access)
    assert.equal(next.refreshToken, 'rt-2')
    assert.equal(next.cliBound, true)
    assert.equal(readDoc(dir).tokens.refresh_token, 'rt-2', 'the CLI file is left as the CLI wrote it')
  })
})

test('synced refresh spends the CLI\'s newer refresh token when its access token is also expiring', async () => {
  const ours = { access: accessToken('a1', 30_000), refresh: 'rt-1' }
  const cli = { access: accessToken('a2', 60_000), refresh: 'rt-2' }
  const dir = codexHome(authDoc(cli))
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    await refreshCodexSynced(session(ours, 30_000), rotatingRefresh('a3', spent), PREEMPT_MS)
    assert.equal(spent[0]?.refreshToken, 'rt-2', 'never the token the CLI already rotated past')
    assert.equal(readDoc(dir).tokens.refresh_token, 'rt-a3')
  })
})

test('synced refresh repairs a CLI file left behind on an older rotation', async () => {
  const stale = { access: accessToken('a0', 1000), refresh: 'rt-0' }
  const ours = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
  const dir = codexHome(authDoc(stale))
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    await refreshCodexSynced(session(ours, 60_000), rotatingRefresh('a2', spent), PREEMPT_MS)
    assert.equal(spent[0]?.refreshToken, 'rt-1', 'our newer login is the one refreshed')
    assert.equal(readDoc(dir).tokens.refresh_token, 'rt-a2', 'the CLI file converges on the new rotation')
  })
})

test('synced refresh leaves a CLI file for another account untouched', async () => {
  const other = { access: accessToken('b1', 3600_000), refresh: 'rt-other', accountId: 'acct-2' }
  const ours = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
  const dir = codexHome(authDoc(other))
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    const next = await refreshCodexSynced(session(ours, 60_000), rotatingRefresh('a2', spent), PREEMPT_MS)
    assert.equal(spent[0]?.refreshToken, 'rt-1')
    assert.equal(next.refreshToken, 'rt-a2')
    assert.deepEqual(readDoc(dir), authDoc(other), 'the other account\'s login is not overwritten')
  })
})

test('synced refresh without a CLI login refreshes standalone and creates nothing', async () => {
  const dir = join(tempDir('codex-absent-'), 'missing-home')
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    const ours = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
    const next = await refreshCodexSynced(session(ours, 60_000), rotatingRefresh('a2', spent), PREEMPT_MS)
    assert.equal(next.refreshToken, 'rt-a2')
    assert.equal(existsSync(dir), false, 'no Codex home is created just to sync with it')
  })
})

const flockAvailable = process.platform !== 'win32' && spawnSync('flock', ['--version']).status === 0

test('synced refresh waits for Codex\'s auth.lock before reading the file', flockAvailable ? {} : { skip: 'flock(1) is not installed' }, async () => {
  const tokens = { access: accessToken('a1', 60_000), refresh: 'rt-1' }
  const dir = codexHome(authDoc(tokens))
  // Hold the lock the way the CLI does around a write, and rotate while holding it.
  const cli = { access: accessToken('a2', 10 * 24 * 3600_000), refresh: 'rt-2' }
  const holder = spawn('flock', ['-x', join(dir, 'auth.lock'), 'sh', '-c', 'echo locked; sleep 1'], { stdio: ['ignore', 'pipe', 'ignore'] })
  await new Promise<void>((resolve) => { holder.stdout.on('data', () => { resolve() }) })
  // The CLI rotates 300 ms into its hold: only a read that waited for the lock can see it.
  setTimeout(() => { writeFileSync(join(dir, 'auth.json'), JSON.stringify(authDoc(cli), null, 2), { mode: 0o600 }) }, 300)
  await withEnv('CODEX_HOME', dir, async () => {
    const spent: CodexSession[] = []
    const next = await refreshCodexSynced(session(tokens, 60_000), rotatingRefresh('never', spent), PREEMPT_MS)
    assert.equal(spent.length, 0, 'the read waited for the lock, so it saw the CLI\'s rotation instead of spending rt-1')
    assert.equal(next.refreshToken, 'rt-2')
  })
})

// ---------------------------------------------------------------------------
// login('codex', 'cli')
// ---------------------------------------------------------------------------

function makeController(readCodexCreds: () => CodexSession | undefined) {
  return new SubscriptionsAuthController(
    new OAuthFlowManager(),
    new DeviceFlowManager(),
    () => {},
    () => undefined,
    {},
    () => undefined,
    undefined,
    {},
    {},
    undefined,
    undefined,
    readCodexCreds,
  )
}

async function inIsolatedHome<T>(run: () => Promise<T>): Promise<T> {
  const home = tempDir('codex-login-dsh-')
  return withEnv('DSH_HOME', home, async () => {
    assert.ok(authFilePath().startsWith(home), 'the session store resolves inside the temp home')
    return run()
  })
}

test('login(codex, cli): the CLI login is imported instantly and stays bound', async () => {
  await inIsolatedHome(async () => {
    const imported = session({ access: accessToken('a1', 3600_000), refresh: 'rt-1' }, 3600_000)
    const controller = makeController(() => imported)
    const { authorizeUrl } = await controller.login('codex', 'cli')
    assert.equal(authorizeUrl, '', 'the import path opens no browser')
    const stored = (await listAccounts('codex'))[0]?.session
    assert.equal(stored?.refreshToken, 'rt-1')
    assert.equal(stored?.cliBound, true)
    assert.equal((await controller.status('codex')).accounts.length, 1)
  })
})

test('login(codex, cli): no CLI login is an error, not a silent browser flow', async () => {
  await inIsolatedHome(async () => {
    const controller = makeController(() => undefined)
    await assert.rejects(controller.login('codex', 'cli'), /codex login/)
    assert.equal((await listAccounts('codex')).length, 0)
  })
})

test('login(codex): the default stays the browser flow even when a CLI login exists', async () => {
  await inIsolatedHome(async () => {
    const controller = makeController(() => session({ access: accessToken('a1', 3600_000), refresh: 'rt-1' }, 3600_000))
    try {
      const { authorizeUrl } = await controller.login('codex')
      assert.match(authorizeUrl, /^https:\/\/auth\.openai\.com\/oauth\/authorize\?/)
      assert.equal((await listAccounts('codex')).length, 0, 'nothing was imported')
    } finally {
      await controller.cancel('codex')
    }
  })
})

// ---------------------------------------------------------------------------
// Claude's binding survives refreshes too
// ---------------------------------------------------------------------------

const claudeFileStore = ((): boolean => {
  const saved = process.env.CLAUDE_CONFIG_DIR
  process.env.CLAUDE_CONFIG_DIR = tempDir('claude-probe-')
  try {
    return readClaudeCodeCredentials() === undefined
  } finally {
    if (saved === undefined) delete process.env.CLAUDE_CONFIG_DIR
    else process.env.CLAUDE_CONFIG_DIR = saved
  }
})()

test('refreshClaudeSynced keeps the account keychain-bound across refreshes', claudeFileStore ? {} : { skip: 'a Claude Code Keychain entry shadows the credentials file on this machine' }, async () => {
  const dir = tempDir('claude-bound-')
  const prior: ClaudeSession = { accessToken: 'at-1', refreshToken: 'rt-1', expiresAt: Date.now() + 1000, scopes: 'user:inference', keychainBound: true }
  writeFileSync(join(dir, '.credentials.json'), JSON.stringify({ claudeAiOauth: prior }), { mode: 0o600 })
  await withEnv('CLAUDE_CONFIG_DIR', dir, async () => {
    const refreshed = await refreshClaudeSynced(prior, async () => ({
      accessToken: 'at-2', refreshToken: 'rt-2', expiresAt: Date.now() + 3600_000, scopes: 'user:inference',
    }))
    assert.equal(refreshed.accessToken, 'at-2')
    assert.equal(refreshed.keychainBound, true, 'a refreshed import keeps syncing with Claude Code')
    const adopted = await refreshClaudeSynced({ ...prior, accessToken: 'at-old' }, async () => {
      throw new Error('a fresh token in Claude Code\'s store needs no refresh')
    })
    assert.equal(adopted.accessToken, 'at-2')
    assert.equal(adopted.keychainBound, true)
  })
})
