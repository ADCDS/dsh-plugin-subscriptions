/**
 * The pool adapter: same-subscription account routing, plus optional
 * configured tier extras. The picker is the union of every account's
 * catalog. A model listed by several accounts failovers; a model listed by
 * one account is pinned to it. Tiers are extra picker rows. Member
 * selection is sticky per session (so prompt caches survive) and optionally
 * quota-aware; failures fail over to the next member as long as no stream
 * content chunk has been emitted (leading usage remains buffered).
 */

import {
  EMPTY_RESPONSE_CODE,
  LlmAdapter,
  LlmError,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmResolvedModelInfo,
  PreparedAdapterCall,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { ProviderId } from '../auth/store.js'
import type { AccountAwareAdapter } from './accounts.js'
import type { ConcretePoolMember, PoolDefinition, PoolMemberRef } from './pool-family.js'
import { poolKey } from './pool-family.js'
import { accountKey, classifyPoolFailure, CODEX_BURST_COOLDOWN_MS, memberKey, PoolHealthRegistry } from './pool-health.js'
import type { MemberQuota, PoolUsageTracker } from './pool-usage.js'
import { withAbortSignal } from './common.js'
import { streamAccountWithReplay } from './replay.js'

/** Member-selection strategy: plain priority failover or quota-aware scheduling. */
export type PoolStrategy = 'priority' | 'quota_aware'

export interface PoolAdapterOptions {
  /** The live subscription adapters, by provider route. */
  adapters: Partial<Record<ProviderId, AccountAwareAdapter>>
  health: PoolHealthRegistry
  usage: PoolUsageTracker
  strategy: PoolStrategy
  /** A challenger must out-urgency the sticky member by this factor to take over. */
  switchMargin: number
  /** The default account of one provider (for config members omitting `account`). */
  defaultAccount: (provider: ProviderId) => Promise<string | undefined>
  /** Resolve legacy account aliases before pool deduplication and cache identity. */
  resolveAccount?: (provider: ProviderId, account: string) => Promise<string>
  /** Account pools (auto-aggregated plus config overrides), resolved lazily. */
  families: () => Promise<Map<string, PoolDefinition>>
  /** User-configured extra picker entries (heterogeneous fallbacks), by pool id. */
  tiers: Record<string, PoolMemberRef[]>
  onWarn: (message: string) => void
  /** Injectable abortable backoff for deterministic pool tests. */
  retryDelay?: (delayMs: number, signal: AbortSignal | undefined) => Promise<void>
}

/** Bound on sticky-session memory; oldest entries evict past it. */
const STICKY_SESSION_LIMIT = 1000

/** Display form of one member (account shown when pinned). */
function memberLabel(member: PoolMemberRef): string {
  return member.account === undefined
    ? `${member.provider}/${member.model}`
    : `${member.provider}/${member.account}/${member.model}`
}

/** How long a pools snapshot is trusted (auth changes invalidate immediately). */
const POOLS_CACHE_TTL_MS = 5_000
/** Bound on Codex retry rounds before model output. */
const CODEX_MAX_ATTEMPTS = 20
const CODEX_MAX_RETRY_DELAY_MS = 20_000

export class PoolAdapter extends LlmAdapter {
  /** sessionId|poolId → member key of the last member that served a chunk. */
  private readonly sticky = new Map<string, string>()
  /** Messages already warned about — configuration diagnostics repeat every request otherwise. */
  private readonly warned = new Set<string>()
  /**
   * Short-lived pools snapshot. `owns()` runs on every resolveModel — the
   * model picker issues one per entry — and pool assembly touches every
   * provider's catalog and account store, so recompute at most this often.
   * Auth changes bump {@link generation} so a stale snapshot cannot land.
   */
  private poolsCache: { at: number; pools: Map<string, PoolDefinition> } | undefined
  private poolsInflight: Promise<Map<string, PoolDefinition>> | undefined
  private generation = 0

  constructor(private readonly options: PoolAdapterOptions) {
    super()
  }

  /** Drop the pools snapshot so the next read reflects the current accounts. */
  invalidate(): void {
    this.generation += 1
    this.poolsCache = undefined
    this.poolsInflight = undefined
  }

  /** Warn once per distinct message (pools() runs on every request). */
  private warnOnce(message: string): void {
    if (this.warned.has(message)) return
    this.warned.add(message)
    this.options.onWarn(message)
  }

  /** Drop members whose adapter is not registered (copy — caller state is shared). */
  private usable(pools: Map<string, PoolDefinition>): Map<string, PoolDefinition> {
    const result = new Map<string, PoolDefinition>(pools)
    for (const [id, definition] of [...result]) {
      const kept = definition.members.filter(member => this.options.adapters[member.provider] !== undefined)
      if (kept.length === 0) result.delete(id)
      else if (kept.length < definition.members.length) result.set(id, { ...definition, members: kept })
    }
    return result
  }

  /** Account pools (auto-aggregated plus config overrides) with usable members. */
  private async familyPools(): Promise<Map<string, PoolDefinition>> {
    return this.usable(new Map<string, PoolDefinition>(await this.options.families()))
  }

  /** All pools (account pools merged with extra tiers) with usable members. */
  private async pools(): Promise<Map<string, PoolDefinition>> {
    const cached = this.poolsCache
    if (cached !== undefined && Date.now() - cached.at < POOLS_CACHE_TTL_MS) return cached.pools
    const gen = this.generation
    this.poolsInflight ??= this.assemblePools()
      .then((pools) => {
        if (this.generation === gen) this.poolsCache = { at: Date.now(), pools }
        return pools
      })
      .finally(() => {
        this.poolsInflight = undefined
      })
    return this.poolsInflight
  }

  /** Recompute the pools snapshot (account pools merged with extra tiers). */
  private async assemblePools(): Promise<Map<string, PoolDefinition>> {
    const pools = await this.familyPools()
    for (const [id, members] of Object.entries(this.options.tiers)) {
      if (members.length === 0) continue
      const owner = members[0].provider
      const key = poolKey(owner, id)
      if (pools.has(key)) this.warnOnce(`tier pool "${id}" overrides the account pool of the same id under ${owner}`)
      pools.set(key, { members, extra: true })
    }
    return this.usable(pools)
  }

  /**
   * Extra picker rows one provider lists (configured tiers). Account pools
   * reuse the catalog entry of the same wire id, so they are not listed
   * again — the picker stays one row per model in ChatGPT / Claude / ….
   */
  async modelsForProvider(provider: ProviderId): Promise<LlmModelInfo[]> {
    const pools = await this.pools()
    const models: LlmModelInfo[] = []
    for (const [key, definition] of pools) {
      if (definition.extra !== true) continue
      if (!key.startsWith(`${provider}/`)) continue
      const id = key.slice(provider.length + 1)
      models.push({
        provider,
        id,
        name: definition.name ?? id,
        ...definition.description === undefined ? {} : { description: definition.description },
      })
    }
    return models
  }

  /**
   * Whether `model` on `provider`'s route is served here (several accounts
   * fail over, one account is pinned, or a configured tier).
   */
  async owns(provider: ProviderId, model: string): Promise<boolean> {
    return (await this.pools()).has(poolKey(provider, model))
  }

  /**
   * Resolve every member's account (config members may omit it to mean "the
   * default account") and drop members with no resolvable login. Duplicates
   * collapse — an explicitly pinned account and the default may coincide.
   */
  private async concrete(members: readonly PoolMemberRef[]): Promise<ConcretePoolMember[]> {
    const seen = new Set<string>()
    const resolved: ConcretePoolMember[] = []
    for (const member of members) {
      const requested = member.account ?? await this.options.defaultAccount(member.provider)
      if (requested === undefined) continue
      const account = await (this.options.resolveAccount?.(member.provider, requested) ?? Promise.resolve(requested))
      const key = memberKey(member.provider, account, member.model)
      if (seen.has(key)) continue
      seen.add(key)
      resolved.push({ provider: member.provider, account, model: member.model })
    }
    return resolved
  }

  /**
   * Resolve a pool model to the conservative INTERSECTION of its members'
   * capabilities: the smallest context window and output cap, the reasoning
   * efforts every member supports, and the modalities all of them accept —
   * so a request valid for the pool stays valid after a failover. Capability
   * metadata is account-specific, so every distinct member is resolved.
   */
  override async resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    return (await this.prepareCall(provider, model, signal)).model
  }

  override async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    const definition = (await withAbortSignal(() => this.pools(), signal)).get(poolKey(provider, model))
    if (definition === undefined) throw new LlmError(`unknown pool model "${model}"`, 'NO_ADAPTER')
    const resolved: LlmResolvedModelInfo[] = []
    let lastFailure: unknown
    const members = await withAbortSignal(() => this.concrete(definition.members), signal)
    const eligible: ConcretePoolMember[] = []
    for (const member of members) {
      const adapter = this.options.adapters[member.provider]
      if (adapter === undefined) continue
      // Tolerate per-member failures (a misconfigured tier member, a
      // logged-out provider throwing AUTH): the pool serves as long as ONE
      // member resolves, mirroring stream()'s failover semantics.
      try {
        resolved.push(await withAbortSignal(() => adapter.resolveOwnModel(member.provider, member.model, member.account), signal))
        eligible.push(member)
      } catch (error: unknown) {
        signal?.throwIfAborted()
        lastFailure = error
        this.warnOnce(
          `pool "${model}": member ${String(members.indexOf(member) + 1)} failed to resolve`
          + ` (${error instanceof LlmError ? error.code : 'UNKNOWN'}); excluding it`,
        )
      }
    }
    if (resolved.length === 0) {
      throw new LlmError(`pool "${model}" has no usable member`, 'NO_ADAPTER', {
        ...lastFailure instanceof LlmError && lastFailure.failure.status !== undefined
          ? { status: lastFailure.failure.status } : {},
      })
    }
    const contextWindows = resolved.map(info => info.context?.contextWindow).filter(isNumber)
    const maxTokens = resolved.map(info => info.defaultMaxTokens).filter(isNumber)
    const reasoning = intersectReasoning(resolved)
    const modalities = intersectModalities(resolved)
    const info: LlmResolvedModelInfo = {
      provider,
      id: model,
      name: definition.name ?? model,
      ...definition.description === undefined ? {} : { description: definition.description },
      ...contextWindows.length > 0 ? { context: { contextWindow: Math.min(...contextWindows) } } : {},
      ...maxTokens.length > 0 ? { defaultMaxTokens: Math.min(...maxTokens) } : {},
      ...reasoning === undefined ? {} : { reasoning },
      ...modalities === undefined ? {} : { inputModalities: modalities },
    }
    return { model: info, stream: options => this.streamMembers(options, eligible) }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const definition = (await withAbortSignal(() => this.pools(), options.signal)).get(poolKey(options.provider, options.model))
    if (definition === undefined) throw new LlmError(`unknown pool model "${options.model}"`, 'NO_ADAPTER')
    const members = await withAbortSignal(() => this.concrete(definition.members), options.signal)
    yield* this.streamMembers(options, members)
  }

  private async *streamMembers(options: GenerateOptions, members: ConcretePoolMember[]): AsyncIterable<StreamChunk> {
    let lastError: unknown
    const failures = new Map<string, string>()
    // Retry pre-output Codex transients before ending a turn or subagent activation.
    // Other providers retain a single pool pass.
    for (let attempt = 0; attempt < (options.provider === 'codex' ? CODEX_MAX_ATTEMPTS : 1); attempt++) {
      if (options.signal?.aborted) throw new LlmError('pool request aborted by caller', 'ABORTED', { cause: options.signal.reason })
      const candidates = await withAbortSignal(() => this.select(options.model, members, options.sessionId), options.signal)
      for (const member of candidates) {
        const adapter = this.options.adapters[member.provider]
        if (adapter === undefined) continue
        const iterator = streamAccountWithReplay(
          adapter, { ...options, provider: member.provider }, member.account, member.model,
        )[Symbol.asyncIterator]()
        let first: IteratorResult<StreamChunk>
        const leadingUsage: StreamChunk[] = []
        try {
          first = await iterator.next()
          while (!first.done && first.value.type === 'usage') {
            leadingUsage.push(first.value)
            first = await iterator.next()
          }
          if (first.done === true) {
            throw new LlmError(`${memberLabel(member)} returned an empty stream`, EMPTY_RESPONSE_CODE)
          }
          if (first.value.type === 'finish' && first.value.reason.kind === 'error') {
            const failure = first.value.reason.failure
            throw new LlmError(failure.message, failure.code, failure)
          }
        } catch (error: unknown) {
          try { await iterator.return?.() } catch { /* preserve the attempt failure */ }
          options.signal?.throwIfAborted()
          const classification = classifyPoolFailure(error, member.provider)
          if (classification.action === 'throw') throw error
          if ('cooldownMs' in classification) {
            this.options.health.markUnavailable(
              classification.scope === 'account'
                ? accountKey(member.provider, member.account)
                : memberKey(member.provider, member.account, member.model),
              classification.cooldownMs,
              classification.reason,
            )
            // Refresh successful usage telemetry after a rejected stream, but
            // do not clear a usage endpoint's own rate-limit cooldown.
            if (classification.reason === QUOTA_EXCEEDED_CODE || classification.reason === 'RATE_LIMIT'
              || classification.reason === 'CODEX_BURST_429') {
              this.options.usage.invalidateSnapshot(member.provider, member.account)
            }
          }
          this.options.onWarn(
            `pool "${options.model}": member ${String(members.indexOf(member) + 1)} failed before any output`
            + ` (${error instanceof LlmError ? error.code : 'UNKNOWN'}); trying the next member`,
          )
          failures.set(memberKey(member.provider, member.account, member.model),
            'cooldownMs' in classification ? classification.reason : error instanceof LlmError ? error.code : 'UNKNOWN')
          lastError = error
          continue
        }
        this.remember(options.model, options.sessionId, member)
        // Past the first chunk there is no clean attempt boundary: whatever
        // the member does next (including failing) reaches the caller as-is.
        // Closing a half-consumed member stream must not mask the outcome.
        try {
          yield* leadingUsage
          yield first.value
          for (let next = await iterator.next(); next.done !== true; next = await iterator.next()) {
            yield next.value
          }
        } finally {
          try {
            await iterator.return?.()
          } catch {
            // Closing a half-consumed member stream must not mask the outcome.
          }
        }
        return
      }
      if (options.provider !== 'codex' || attempt + 1 >= CODEX_MAX_ATTEMPTS) break
      const now = Date.now()
      const eligible = members.flatMap(member => {
        const key = memberKey(member.provider, member.account, member.model)
        const reason = failures.get(key)
        if (reason === undefined || !['CODEX_BURST_429', 'SERVER', 'TIMEOUT', 'EMPTY_RESPONSE', 'TRANSPORT'].includes(reason)) return []
        const cooldown = this.options.health.memberCooldown(member.provider, member.account, member.model, now)
        // A later account-wide quota/auth cooldown overrides a prior transient.
        if (cooldown !== undefined && !['CODEX_BURST_429', 'SERVER', 'TIMEOUT', 'EMPTY_RESPONSE'].includes(cooldown.reason)) return []
        const readyAt = cooldown?.until ?? now
        // Never wait out a provider-disclosed long reset inside pool retries.
        return readyAt - now <= CODEX_BURST_COOLDOWN_MS ? [{ readyAt }] : []
      })
      if (eligible.length === 0) break
      const earliest = Math.min(...eligible.map(entry => entry.readyAt))
      const backoffMs = Math.min(CODEX_MAX_RETRY_DELAY_MS, CODEX_BURST_COOLDOWN_MS * 2 ** Math.min(attempt, 10))
      await (this.options.retryDelay ?? poolRetryDelay)(Math.max(backoffMs, earliest - Date.now()), options.signal)
      continue
    }
    throw this.exhausted(options.model, members, lastError)
  }

  /**
   * Order the candidates for one request. Health filters both strategies;
   * `quota_aware` then ranks by urgency (members without telemetry, e.g.
   * copilot, score zero and sink to the bottom of their class), while
   * quota-exhausted members stay as a last-resort tail in pool order. The
   * sticky member keeps its lead unless a challenger out-scores it by
   * `switchMargin`.
   */
  private async select(
    poolId: string,
    members: ConcretePoolMember[],
    sessionId: GenerateOptions['sessionId'],
  ): Promise<ConcretePoolMember[]> {
    const usable = members.filter(member =>
      this.options.adapters[member.provider] !== undefined
      && this.options.health.isMemberAvailable(member.provider, member.account, member.model))
    if (usable.length === 0) return []
    const stickyMember = sessionId === undefined
      ? undefined
      : usable.find(member =>
        memberKey(member.provider, member.account, member.model) === this.sticky.get(stickyKey(poolId, sessionId)))
    if (this.options.strategy === 'priority') {
      return stickyMember === undefined
        ? usable
        : [stickyMember, ...usable.filter(member => member !== stickyMember)]
    }
    const quotas = new Map<ConcretePoolMember, MemberQuota>(
      await Promise.all(usable.map(async member => [member, await this.options.usage.quotaFor(member)] as const)),
    )
    const scored = usable.filter(member => quotas.get(member)?.available === true)
    const quotaFull = usable.filter(member => quotas.get(member)?.available === false)
    scored.sort((a, b) => (quotas.get(b)?.urgency ?? 0) - (quotas.get(a)?.urgency ?? 0))
    if (stickyMember !== undefined && scored.includes(stickyMember)) {
      const best = scored[0]
      const stickyUrgency = quotas.get(stickyMember)?.urgency ?? 0
      const bestUrgency = quotas.get(best)?.urgency ?? 0
      if (best === stickyMember || bestUrgency <= stickyUrgency * this.options.switchMargin) {
        // Sticky holds (no challenger beats it by the margin): lead with it.
        scored.splice(scored.indexOf(stickyMember), 1)
        scored.unshift(stickyMember)
      }
    }
    return [...scored, ...quotaFull]
  }

  /** Pin the serving member to the session (with bounded memory). */
  private remember(poolId: string, sessionId: GenerateOptions['sessionId'], member: ConcretePoolMember): void {
    if (sessionId === undefined) return
    const key = stickyKey(poolId, sessionId)
    this.sticky.delete(key)
    if (this.sticky.size >= STICKY_SESSION_LIMIT) {
      const oldest = this.sticky.keys().next()
      if (oldest.done !== true) this.sticky.delete(oldest.value)
    }
    this.sticky.set(key, memberKey(member.provider, member.account, member.model))
  }

  /** Earliest time a member clears both its own and its account's cooldown. */
  private recovery(pool: ConcretePoolMember[], now = Date.now()): { until: number; reason: string } | undefined {
    let first: { until: number; reason: string } | undefined
    for (const member of pool) {
      const cooling = this.options.health.memberCooldown(member.provider, member.account, member.model, now)
      if (cooling !== undefined && (first === undefined || cooling.until < first.until)) first = cooling
    }
    return first
  }

  /** Report safe per-member cooldown reasons without exposing account identities or upstream bodies. */
  private exhausted(model: string, pool: ConcretePoolMember[], cause?: unknown): LlmError {
    const now = Date.now()
    const cooldowns = pool.map(member => this.options.health.memberCooldown(member.provider, member.account, member.model, now))
    const states = cooldowns.map((cooling, index) =>
      `${String(index + 1)}:${cooling === undefined ? 'not cooling' : `${cooling.reason} (${cooling.scope}, ${String(Math.ceil((cooling.until - now) / 1_000))}s)`}`)
    const recovery = this.recovery(pool, now)
    const retryAfterMs = recovery === undefined ? undefined : Math.max(recovery.until - now, 1)
    const reasons = cooldowns.map(cooling => cooling?.reason)
    const reason = recovery?.reason
    const code = cause instanceof LlmError ? cause.code : pool.length === 0 ? 'NO_ADAPTER'
      : reasons.every(item => item === 'AUTH' || item === 'INVALID_CREDENTIAL' || item === 'MISSING_CREDENTIAL') ? 'AUTH'
        : reasons.some(item => item === 'RATE_LIMIT' || item === 'CODEX_BURST_429' || item === QUOTA_EXCEEDED_CODE)
          ? 'RATE_LIMIT' : reason ?? 'NO_ADAPTER'
    const message = `pool "${model}" exhausted: ${pool.length === 0 ? 'no eligible account' : `members [${states.join(', ')}]`}`
      + `${cause instanceof LlmError ? `; last failure ${cause.code}${cause.failure.status === undefined ? '' : ` (HTTP ${String(cause.failure.status)})`}` : ''}`
    this.options.onWarn(message)
    return new LlmError(
      message,
      code,
      { ...retryAfterMs === undefined || code === 'AUTH' || code === 'NO_ADAPTER'
        ? {} : { providerRetryAfterMs: retryAfterMs },
        ...cause instanceof LlmError && cause.failure.status !== undefined ? { status: cause.failure.status } : {},
        ...cause instanceof LlmError && cause.failure.requestId !== undefined ? { requestId: cause.failure.requestId } : {},
      },
    )
  }
}

/** Abortable local backoff between Codex pool rounds; no provider-reported long reset is waited here. */
function poolRetryDelay(delayMs: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) { reject(new LlmError('pool request aborted by caller', 'ABORTED')); return }
    const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve() }, delayMs)
    function onAbort(): void { clearTimeout(timer); reject(new LlmError('pool request aborted by caller', 'ABORTED')) }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

function stickyKey(poolId: string, sessionId: NonNullable<GenerateOptions['sessionId']>): string {
  return `${String(sessionId)}|${poolId}`
}

function isNumber(value: number | undefined): value is number {
  return value !== undefined
}

/** Reasoning efforts every member supports (id intersection, first member's order). */
function intersectReasoning(
  resolved: readonly LlmResolvedModelInfo[],
): LlmResolvedModelInfo['reasoning'] | undefined {
  const [first, ...rest] = resolved
  if (first?.reasoning === undefined) return undefined
  const efforts = first.reasoning.efforts.filter(effort =>
    rest.every(info => info.reasoning?.efforts.some(other => other.id === effort.id) === true))
  if (efforts.length === 0) return undefined
  const defaultEffort = first.reasoning.defaultEffort !== undefined
    && efforts.some(effort => effort.id === first.reasoning?.defaultEffort)
    ? first.reasoning.defaultEffort
    : undefined
  return { efforts, ...defaultEffort === undefined ? {} : { defaultEffort } }
}

/** Modalities all members accept; undefined when any member leaves it unknown. */
function intersectModalities(
  resolved: readonly LlmResolvedModelInfo[],
): LlmResolvedModelInfo['inputModalities'] | undefined {
  const [first, ...rest] = resolved
  if (first?.inputModalities === undefined) return undefined
  const modalities = first.inputModalities.filter(modality =>
    rest.every(info => info.inputModalities?.includes(modality) === true))
  // An empty intersection would declare negative capability ("accepts
  // nothing"); report unknown instead — the serving member enforces its own
  // limits at request time.
  return modalities.length === 0 ? undefined : modalities
}
