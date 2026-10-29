/**
 * Read-only subscription usage feed for local tools (status bars, hardware
 * displays) that are not browsers and therefore cannot hold the dsh web
 * session cookie.
 *
 * The feed is a separate loopback HTTP server, off by default. It answers one
 * route, `GET /usage`, with every logged-in account's usage windows, served
 * through the same usage path as the Settings page (the pool's cached,
 * negatively-cached, refresh-aware snapshots), so polling the feed adds no
 * upstream traffic beyond what that cache already allows.
 *
 * Callers authenticate with a bearer token read from a 0600 file in the
 * plugin's data directory; the token file is created on first start. The
 * feed exposes usage percentages, reset times, plan names and account labels
 * only — never tokens.
 */

import { randomBytes, timingSafeEqual } from 'node:crypto'
import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { dirname } from 'node:path'
import { dshHomePath } from '@deepseek-ai/dsh-home-paths'
import type { ProviderId } from './auth/store.js'
import type { ProviderUsage } from './providers/common.js'

/** Feed configuration as the plugin config accepts it. */
export interface UsageFeedConfig {
  /** Serve the feed (default false). */
  enabled?: boolean
  /** Loopback address to bind (default 127.0.0.1). */
  host?: string
  /** TCP port to bind (default 8771). */
  port?: number
}

/** One account as the feed reports it. */
export interface UsageFeedAccount {
  /** Stable account key used by the plugin's auth store. */
  key: string
  /** Human-readable account label (email or workspace), when known. */
  account?: string
  /** Plan name from the stored session, when known. */
  plan?: string
  /** Whether this is the provider's default account. */
  isDefault: boolean
  /** The usage snapshot, when the lookup succeeded. */
  usage?: ProviderUsage
  /** The lookup failure, when it failed. */
  error?: string
}

/** Body of `GET /usage`. */
export interface UsageFeedBody {
  /** Epoch milliseconds at which the feed assembled this body. */
  generatedAt: number
  /** Accounts per configured provider. */
  providers: Partial<Record<ProviderId, UsageFeedAccount[]>>
}

/** Account listing and usage lookup the feed reads from. */
export interface UsageFeedSource {
  /** Providers the plugin serves. */
  readonly providers: readonly ProviderId[]
  /**
   * List one provider's accounts, default first.
   * @param provider - the provider to list.
   * @returns the accounts without credentials.
   */
  accounts(provider: ProviderId): Promise<Array<Omit<UsageFeedAccount, 'usage' | 'error'>>>
  /**
   * Read one account's usage through the cached usage path.
   * @param provider - the account's provider.
   * @param key - the account key.
   * @param signal - aborts the lookup.
   * @returns the usage snapshot.
   */
  usage(provider: ProviderId, key: string, signal: AbortSignal): Promise<ProviderUsage>
}

/** Default loopback port for the feed. */
export const USAGE_FEED_DEFAULT_PORT = 8771
/** Upper bound on assembling one response; a hung upstream must not hang the client. */
const FEED_TIMEOUT_MS = 15_000
const TOKEN_BYTES = 32
const LOOPBACK_HOSTS = new Set(['127.0.0.1', '::1', 'localhost'])

/**
 * Path of the bearer token file.
 * @returns `dshHomePath('plugins', 'subscriptions', 'usage-feed.token')`.
 */
export function usageFeedTokenPath(): string {
  return dshHomePath('plugins', 'subscriptions', 'usage-feed.token')
}

/**
 * Resolve and validate the feed configuration. The feed binds loopback only:
 * it serves account labels, and loopback is its only access control besides
 * the token.
 * @param config - raw config section, when present.
 * @param path - diagnostic path naming the config that owns the value.
 * @returns the resolved configuration, or undefined when the feed is disabled.
 */
export function resolveUsageFeed(
  config: UsageFeedConfig | undefined,
  path: string,
): { host: string; port: number } | undefined {
  if (config?.enabled !== true) return undefined
  const host = config.host ?? '127.0.0.1'
  if (!LOOPBACK_HOSTS.has(host)) {
    throw new Error(`${path}.host must be a loopback address (127.0.0.1, ::1 or localhost), got "${host}"`)
  }
  const port = config.port ?? USAGE_FEED_DEFAULT_PORT
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${path}.port must be an integer between 1 and 65535, got ${String(port)}`)
  }
  return { host, port }
}

/**
 * Read the bearer token, creating a random one (mode 0600) when absent.
 * @param path - token file path.
 * @returns the token.
 */
export async function loadOrCreateToken(path = usageFeedTokenPath()): Promise<string> {
  try {
    const existing = (await readFile(path, 'utf8')).trim()
    if (existing.length >= 32) return existing
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
  }
  const token = randomBytes(TOKEN_BYTES).toString('base64url')
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  await writeFile(path, `${token}\n`, { mode: 0o600 })
  // writeFile's mode only applies on creation; an existing short file keeps its mode.
  await chmod(path, 0o600)
  return token
}

function bearerMatches(header: string | undefined, token: string): boolean {
  if (header === undefined || !header.startsWith('Bearer ')) return false
  const actual = Buffer.from(header.slice('Bearer '.length).trim(), 'utf8')
  const expected = Buffer.from(token, 'utf8')
  return actual.byteLength === expected.byteLength && timingSafeEqual(actual, expected)
}

/**
 * Assemble the feed body. Providers and accounts are looked up concurrently;
 * one account's failure becomes that account's `error` and never fails the body.
 * @param source - account listing and usage lookup.
 * @param signal - aborts outstanding lookups.
 * @returns the assembled body.
 */
export async function collectUsage(source: UsageFeedSource, signal: AbortSignal): Promise<UsageFeedBody> {
  const entries = await Promise.all(source.providers.map(async (provider) => {
    let accounts: Array<Omit<UsageFeedAccount, 'usage' | 'error'>>
    try {
      accounts = await source.accounts(provider)
    } catch (error) {
      return [provider, [{ key: '', isDefault: false, error: messageOf(error) }]] as const
    }
    const resolved = await Promise.all(accounts.map(async (account): Promise<UsageFeedAccount> => {
      try {
        return { ...account, usage: await source.usage(provider, account.key, signal) }
      } catch (error) {
        return { ...account, error: messageOf(error) }
      }
    }))
    return [provider, resolved] as const
  }))
  return { generatedAt: Date.now(), providers: Object.fromEntries(entries) }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function send(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(text),
  })
  res.end(text)
}

/**
 * Handle one feed request.
 * @param req - incoming request.
 * @param res - response to write.
 * @param token - expected bearer token.
 * @param source - account listing and usage lookup.
 */
export async function handleFeedRequest(
  req: IncomingMessage,
  res: ServerResponse,
  token: string,
  source: UsageFeedSource,
): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://feed.invalid').pathname
  if (path !== '/usage') {
    send(res, 404, { error: 'not found' })
    return
  }
  if (req.method !== 'GET') {
    res.setHeader('allow', 'GET')
    send(res, 405, { error: 'method not allowed' })
    return
  }
  // Browsers attach Origin to cross-site requests; a local tool does not. The
  // token is the credential, but refusing browser origins keeps a page from
  // probing the feed even if it learns the token.
  if (req.headers.origin !== undefined) {
    send(res, 403, { error: 'browser requests are not accepted' })
    return
  }
  if (!bearerMatches(req.headers.authorization, token)) {
    send(res, 401, { error: 'missing or invalid bearer token' })
    return
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FEED_TIMEOUT_MS)
  res.on('close', () => controller.abort())
  try {
    send(res, 200, await collectUsage(source, controller.signal))
  } finally {
    clearTimeout(timer)
  }
}

/**
 * Start the feed server.
 * @param address - resolved loopback address and port.
 * @param source - account listing and usage lookup.
 * @param onWarn - receives a bind or request failure.
 * @param tokenPath - bearer token file path.
 * @returns a function that closes the server.
 */
export async function startUsageFeed(
  address: { host: string; port: number },
  source: UsageFeedSource,
  onWarn: (message: string) => void,
  tokenPath = usageFeedTokenPath(),
): Promise<() => Promise<void>> {
  const token = await loadOrCreateToken(tokenPath)
  const server: Server = createServer((req, res) => {
    handleFeedRequest(req, res, token, source).catch((error: unknown) => {
      onWarn(`usage feed request failed: ${messageOf(error)}`)
      if (!res.headersSent) send(res, 500, { error: 'internal error' })
      else res.destroy()
    })
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(address.port, address.host, () => {
      server.off('error', reject)
      resolve()
    })
  })
  server.on('error', error => onWarn(`usage feed server error: ${messageOf(error)}`))
  return () => new Promise<void>((resolve) => {
    server.closeAllConnections()
    server.close(() => resolve())
  })
}
