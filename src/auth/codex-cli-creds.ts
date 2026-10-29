/**
 * Codex CLI credential import: read the ChatGPT login the `codex` CLI keeps in
 * `$CODEX_HOME/auth.json` (default `~/.codex`), and keep that file in step with
 * the refreshes this plugin performs for an imported account.
 *
 * OpenAI rotates refresh tokens and rejects a reused one, so two clients that
 * refresh one login independently would log each other out. Codex itself
 * re-reads `auth.json` under its refresh guard before refreshing and adopts a
 * token another process rotated, so the contract here is the mirror image:
 * before refreshing, adopt a newer rotation from the file; after refreshing,
 * write the new tokens back. The whole read-refresh-write runs under Codex's
 * own `auth.lock` (flock) when `flock(1)` is available, and the write is a
 * compare-and-swap on the file's access token either way.
 */
import { spawn } from 'node:child_process'
import { chmodSync, existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { decodeJwtPayload } from './jwt.js'
import type { CodexSession } from './store.js'
import { CODEX_CLIENT_ID, codexProfileClaims } from '../providers/codex.js'

/** How long to wait for Codex's `auth.lock` before falling back to compare-and-swap alone. */
const LOCK_WAIT_SECONDS = 10

interface CodexAuthTokens {
  id_token?: unknown
  access_token?: unknown
  refresh_token?: unknown
  account_id?: unknown
}

/** The subset of Codex's `auth.json` this module reads; every other field is preserved on write. */
interface CodexAuthJson {
  auth_mode?: unknown
  tokens?: CodexAuthTokens | null
  last_refresh?: unknown
  [field: string]: unknown
}

/** `$CODEX_HOME`, defaulting to `~/.codex` like the CLI. */
function codexHome(): string {
  return process.env.CODEX_HOME ?? join(homedir(), '.codex')
}

function authFilePath(): string {
  return join(codexHome(), 'auth.json')
}

function nonEmpty(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function readAuthJson(): CodexAuthJson | undefined {
  let raw: string
  try {
    raw = readFileSync(authFilePath(), 'utf8')
  } catch {
    return undefined
  }
  try {
    const parsed = JSON.parse(raw) as unknown
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) ? parsed as CodexAuthJson : undefined
  } catch {
    return undefined
  }
}

/** `chatgpt_account_id` from the id token, the same claim the browser login reads. */
function accountIdOfIdToken(idToken: string): string | undefined {
  const auth = decodeJwtPayload(idToken)?.['https://api.openai.com/auth']
  return typeof auth === 'object' && auth !== null
    ? nonEmpty((auth as Record<string, unknown>).chatgpt_account_id)
    : undefined
}

/** Convert one `auth.json` document into a session, or `undefined` when it holds no usable ChatGPT login. */
function toSession(doc: CodexAuthJson): CodexSession | undefined {
  // API-key logins carry no OAuth tokens to share; older files omit auth_mode.
  if (doc.auth_mode !== undefined && doc.auth_mode !== 'chatgpt') return undefined
  const provenance = doc.tokens === null || doc.tokens === undefined
    ? undefined
    : (doc.tokens as Record<string, unknown>).oauth_provenance
  // A login issued to another OAuth client cannot be refreshed with this plugin's client id.
  const clientId = typeof provenance === 'object' && provenance !== null
    ? (provenance as Record<string, unknown>).client_id
    : undefined
  if (clientId !== undefined && clientId !== CODEX_CLIENT_ID) return undefined
  const tokens = doc.tokens ?? undefined
  const accessToken = nonEmpty(tokens?.access_token)
  const refreshToken = nonEmpty(tokens?.refresh_token)
  const idToken = nonEmpty(tokens?.id_token)
  if (accessToken === undefined || refreshToken === undefined || idToken === undefined) return undefined
  const exp = decodeJwtPayload(accessToken)?.exp
  if (typeof exp !== 'number' || !Number.isFinite(exp) || exp <= 0) return undefined
  const accountId = nonEmpty(tokens?.account_id) ?? accountIdOfIdToken(idToken)
  if (accountId === undefined) return undefined
  return {
    accessToken,
    refreshToken,
    expiresAt: Math.trunc(exp * 1000),
    accountId,
    idToken,
    ...codexProfileClaims(idToken),
    cliBound: true,
  }
}

/**
 * Read the Codex CLI's current ChatGPT login.
 * @returns the session, marked CLI-bound, or `undefined` when the CLI holds no usable login.
 */
export function readCodexCliCredentials(): CodexSession | undefined {
  const doc = readAuthJson()
  return doc === undefined ? undefined : toSession(doc)
}

/**
 * Write a refreshed session into the CLI's `auth.json`, preserving every field
 * this module does not own, when the file still holds the login the refresh
 * started from (same account, access token `expectedAccessToken`). The write
 * replaces the file atomically with mode 0600.
 * @param next - the freshly refreshed session.
 * @param expectedAccessToken - the file's access token when the refresh began.
 * @returns whether the file was updated.
 */
export function writeBackCodexCliCredentials(next: CodexSession, expectedAccessToken: string): boolean {
  const doc = readAuthJson()
  if (doc === undefined) return false
  const current = toSession(doc)
  if (current === undefined || current.accountId !== next.accountId || current.accessToken !== expectedAccessToken) {
    return false
  }
  const tokens = { ...doc.tokens } as Record<string, unknown>
  tokens.access_token = next.accessToken
  tokens.refresh_token = next.refreshToken
  if (next.idToken !== undefined) tokens.id_token = next.idToken
  const updated = { ...doc, tokens, last_refresh: new Date().toISOString() }
  const path = authFilePath()
  const temp = `${path}.dsh-${String(process.pid)}.tmp`
  try {
    writeFileSync(temp, JSON.stringify(updated, null, 2), { encoding: 'utf8', mode: 0o600 })
    chmodSync(temp, 0o600)
    renameSync(temp, path)
    return true
  } catch {
    return false
  }
}

/**
 * Hold Codex's `auth.lock` (the flock its storage takes around every
 * read-modify-write) while `run` executes. `flock(1)` takes the lock in a child
 * process that keeps it until its stdin closes; without `flock(1)`, or when the
 * lock stays busy, `run` proceeds and the compare-and-swap write is the guard.
 */
async function withCodexAuthLock<T>(run: () => Promise<T>): Promise<T> {
  const holder = await acquireAuthLock()
  try {
    return await run()
  } finally {
    holder?.stdin?.end()
  }
}

function acquireAuthLock(): Promise<ReturnType<typeof spawn> | undefined> {
  // Without a Codex home there is no CLI to coordinate with; never create one just to lock it.
  if (process.platform === 'win32' || !existsSync(codexHome())) return Promise.resolve(undefined)
  return new Promise((resolve) => {
    let settled = false
    const settle = (value: ReturnType<typeof spawn> | undefined): void => {
      if (settled) return
      settled = true
      resolve(value)
    }
    let child: ReturnType<typeof spawn>
    try {
      child = spawn('flock', ['-x', '-w', String(LOCK_WAIT_SECONDS), join(codexHome(), 'auth.lock'),
        'sh', '-c', 'echo locked; exec cat >/dev/null'], { stdio: ['pipe', 'pipe', 'ignore'] })
    } catch {
      settle(undefined)
      return
    }
    child.on('error', () => { settle(undefined) })
    child.on('exit', () => { settle(undefined) })
    child.stdout?.on('data', (chunk: Buffer) => {
      if (chunk.toString().includes('locked')) settle(child)
    })
  })
}

/**
 * Refresh a CLI-bound Codex session in step with the Codex CLI. Under the CLI's
 * lock: adopt a newer rotation already in `auth.json` instead of spending a
 * refresh token; otherwise refresh from whichever of the two holds the newer
 * login and write the result back, so the CLI never refreshes with a token this
 * plugin already spent.
 * @param session - the session the token manager wants refreshed.
 * @param doRefresh - the refresh-token grant (network call).
 * @param preemptMs - how long before expiry a session no longer counts as fresh.
 * @returns the freshest session, still marked CLI-bound.
 */
export async function refreshCodexSynced(
  session: CodexSession,
  doRefresh: (session: CodexSession) => Promise<CodexSession>,
  preemptMs: number,
): Promise<CodexSession> {
  return withCodexAuthLock(async () => {
    const source = readCodexCliCredentials()
    const sameLogin = source !== undefined && source.accountId === session.accountId
    const sourceNewer = sameLogin && source.accessToken !== session.accessToken && source.expiresAt > session.expiresAt
    if (sourceNewer && source.expiresAt - Date.now() > preemptMs) return source
    const base = sourceNewer ? source : session
    const next = { ...await doRefresh(base), cliBound: true }
    if (sameLogin) writeBackCodexCliCredentials(next, source.accessToken)
    return next
  })
}
