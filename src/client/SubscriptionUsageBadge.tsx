/**
 * Subscription usage badge: a stats pill in the composer's dock
 * (`conversation.composer.dock`), modelled on the host's own token-usage
 * pill. Collapsed, it shows the rate-limit windows of the provider behind the
 * session's CURRENT model (a GPT model → Codex usage, a Claude model → Claude
 * usage); clicking it opens a trigger-anchored dialog listing every logged-in
 * provider's default account and all of its windows, the current one first.
 *
 * Usage rides the `subscriptions-auth` `status` + `usage` endpoints on a slow
 * poll (the server shares its cache across UI surfaces); the current model
 * comes from ui-model-selection's `modelDirectories` service on a quicker
 * poll, since the host pushes nothing on a model switch. Renders nothing when
 * no provider has a logged-in account that reports usage.
 *
 * Only the default account is shown — the same account direct (non-pool)
 * routes serve; the full per-account breakdown lives in Settings → 订阅.
 * Every color resolves through a `--dsw-*` design token and every
 * user-visible string goes through the locale `t` of the
 * 'settings.subscriptions' namespace.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import type { PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { ConnectionHandle } from '@deepseek-ai/dsh-api-remotes/client'
import { IconDataOutline16, useAnchoredPosition, useDismissOnOutsidePointer } from '@deepseek-ai/dsh-client-ui-primitives'
import { callSubscriptionsAuth, usageBarColor } from './SubscriptionsSection.js'
import type { AccountStatus, ProviderStatus, ProviderUsage, SubscriptionProvider, UsageWindow } from './SubscriptionsSection.js'
import type { ModelDirectoriesLike } from './SpeedSelect.js'
import { en } from './locales.js'
import type { SubscriptionsKey } from './locales.js'

/** How often the badge re-reads usage; the server also shares its own cache/negative-cache across UI surfaces. */
const USAGE_POLL_INTERVAL_MS = 15 * 60_000

/** How often the badge re-reads the session's current model (model switches arrive only by asking). */
const MODEL_POLL_INTERVAL_MS = 3000

/** Distance between the trigger's top edge and the dialog's bottom (host stat dialogs use the same). */
const PANEL_GAP = 8

/** Distance kept between the dialog and each viewport edge. */
const PANEL_MARGIN = 12

/** Injected dependencies (slot `inject`, session-bound). */
export interface SubscriptionUsageBadgeInjected {
  /** Connection RPC caller to reach the `subscriptions-auth` endpoints. */
  rpc: ConnectionHandle['rpc']
  /** Resolve the provider id behind the session's current model; undefined when unknown. */
  currentProvider: () => Promise<string | undefined>
}

/** Props delivered by the slot outlet + inject + the locale seat. */
export type SubscriptionUsageBadgeProps = PropsRuntime<'conversation.composer.dock'>
  & Partial<SubscriptionUsageBadgeInjected>
  & Partial<PropsLocale<'settings.subscriptions'>>

/** One provider's usage snapshot as rendered by the badge (default account only). */
export interface ProviderUsageDisplay {
  provider: SubscriptionProvider
  name: string
  /** Default account's display handle (email / login), when the provider reports one. */
  account?: string
  plan?: string
  windows: UsageWindow[]
}

/** Brand display names (short form for the compact badge). */
const PROVIDER_NAMES: Record<SubscriptionProvider, string> = {
  codex: 'Codex',
  claude: 'Claude',
  grok: 'Grok',
  copilot: 'Copilot',
}

/**
 * The `currentProvider` half of the inject face: the session's effective
 * model provider through ui-model-selection's `modelDirectories` service,
 * resolved lazily per call (the service may register after this plugin, and
 * a shell without it simply reports "unknown", which the badge treats as
 * "show every provider").
 */
export function createCurrentProviderReader(
  models: () => ModelDirectoriesLike | undefined,
  sessionId: string,
): SubscriptionUsageBadgeInjected['currentProvider'] {
  return async () => {
    const directories = models()
    if (directories === undefined) return undefined
    const { current } = await directories.directoryFor(sessionId).load()
    return current?.provider
  }
}

/**
 * Compact time-remaining label derived from the window's `resetsAt` timestamp:
 * "6d18h" (days+hours), "1h58m" (hours+minutes), or "42m" (minutes only).
 * Falls back to the scope/kind abbreviation when no reset time is known.
 */
export function windowLabel(w: UsageWindow): string {
  if (w.resetsAt === undefined) {
    if (w.scope !== undefined && w.scope !== '') return w.scope
    switch (w.kind) {
      case 'session': return '5h'
      case 'weekly': return 'Wk'
      default: return 'W'
    }
  }
  const ms = Math.max(0, w.resetsAt - Date.now())
  const minutes = Math.floor(ms / 60_000)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  if (days > 0) return `${days}d${hours % 24}h`
  if (hours > 0) return `${hours}h${minutes % 60}m`
  return `${Math.max(1, minutes)}m`
}

/** Clamp and round a window's used share for display. */
function usedPercent(w: UsageWindow): number {
  return Math.round(Math.min(100, Math.max(0, w.usedPercent)))
}

/** Compact one-line readout of a provider: `Codex 6d1h 25% · 1h58m 13%`. */
export function compactSegment(d: ProviderUsageDisplay): string {
  const parts = d.windows.map(w => `${windowLabel(w)} ${usedPercent(w)}%`)
  return `${d.name} ${parts.join(' · ')}`
}

/**
 * Pick what the collapsed pill shows: the current model's provider when its
 * usage is known, otherwise every provider (unknown model, a provider this
 * plugin does not serve, or a current provider with no usage to report).
 */
export function collapsedDisplays(
  displays: readonly ProviderUsageDisplay[],
  current: string | undefined,
): readonly ProviderUsageDisplay[] {
  const match = displays.find(d => d.provider === current)
  return match === undefined ? displays : [match]
}

/** Order for the expanded dialog: the current provider first, the rest in poll order. */
export function expandedDisplays(
  displays: readonly ProviderUsageDisplay[],
  current: string | undefined,
): readonly ProviderUsageDisplay[] {
  const match = displays.find(d => d.provider === current)
  return match === undefined ? displays : [match, ...displays.filter(d => d !== match)]
}

/** The default account of a provider's account list, when any is logged in. */
function defaultAccountOf(status: ProviderStatus | undefined): AccountStatus | undefined {
  if (status === undefined || status.accounts.length === 0) return undefined
  return status.accounts.find(a => a.isDefault) ?? status.accounts[0]
}

/** English-dictionary fallback for a missing inject `t` (standalone renders). */
function fallbackTranslate(key: SubscriptionsKey, params?: Record<string, unknown>): string {
  return en[key].replace(/\{(\w+)\}/g, (_, name: string) => String(params?.[name] ?? ''))
}

type Translate = (key: SubscriptionsKey, params?: Record<string, unknown>) => string

/** Localized label of one usage window (kind, plus the model scope when named). */
function usageWindowLabel(t: Translate, window: UsageWindow): string {
  const base = window.kind === 'session'
    ? t('usageSession')
    : window.kind === 'weekly' ? t('usageWeekly') : t('usageWindow')
  return window.scope !== undefined && window.scope !== '' ? `${base} · ${window.scope}` : base
}

/**
 * The composer subscription-usage badge: a pill reading e.g.
 * `Codex 6d1h 25%` for the current model's provider, opening a dialog with
 * every provider's default-account windows. Returns null when no data is
 * available.
 */
export function SubscriptionUsageBadge({ rpc, currentProvider, t }: SubscriptionUsageBadgeProps) {
  const translate: Translate = t ?? fallbackTranslate
  const [displays, setDisplays] = useState<ProviderUsageDisplay[]>([])
  const [current, setCurrent] = useState<string | undefined>(undefined)
  const [open, setOpen] = useState(false)
  const [hover, setHover] = useState(false)
  const inflightRef = useRef(false)
  const mountedRef = useRef(true)
  const rootRef = useRef<HTMLSpanElement | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  // The inject face may be re-evaluated (new callback identities) on
  // re-render; the model poll mounts once and reads through this ref.
  const currentRef = useRef(currentProvider)
  currentRef.current = currentProvider
  // Last-known-good display per provider, kept across a failed poll (e.g. a
  // 429 during the server's own negative-cache cooldown) so the segment
  // doesn't flicker away — it only disappears once the default account
  // actually logs out or a fetch succeeds but reports the window as
  // unsupported.
  const lastKnownRef = useRef(new Map<SubscriptionProvider, ProviderUsageDisplay>())

  const refresh = useCallback(async (): Promise<void> => {
    if (rpc === undefined || inflightRef.current) return
    inflightRef.current = true
    try {
      const statusResp = await callSubscriptionsAuth<{
        providers: Record<SubscriptionProvider, ProviderStatus>
      }>(rpc, 'status', {})
      if (!mountedRef.current) return

      const accounts = new Map<SubscriptionProvider, AccountStatus>()
      for (const id of Object.keys(statusResp.providers) as SubscriptionProvider[]) {
        const account = defaultAccountOf(statusResp.providers[id])
        if (account !== undefined) accounts.set(id, account)
      }

      const lastKnown = lastKnownRef.current
      // Drop last-known state for anything no longer logged in — that is a
      // real signal, unlike a fetch failure.
      for (const provider of lastKnown.keys()) {
        if (!accounts.has(provider)) lastKnown.delete(provider)
      }

      if (accounts.size === 0) {
        setDisplays([])
        return
      }

      const results = await Promise.allSettled(
        [...accounts.entries()].map(async ([provider, account]) => {
          const usage = await callSubscriptionsAuth<ProviderUsage>(rpc, 'usage', { provider, account: account.key })
          return { provider, account, usage }
        }),
      )
      if (!mountedRef.current) return

      for (const r of results) {
        if (r.status !== 'fulfilled') continue // keep whatever is cached for this provider
        const { provider, account, usage } = r.value
        if (!usage.supported || !usage.windows || usage.windows.length === 0) {
          lastKnown.delete(provider)
          continue
        }
        lastKnown.set(provider, {
          provider,
          name: PROVIDER_NAMES[provider],
          ...account.account === undefined ? {} : { account: account.account },
          ...(usage.plan ?? account.plan) === undefined ? {} : { plan: usage.plan ?? account.plan },
          windows: usage.windows,
        })
      }
      // Render in a stable order (the account map's insertion order, which
      // follows Object.keys(statusResp.providers)) so a segment doesn't jump
      // around as polls settle at different times.
      const order = [...accounts.keys()]
      setDisplays(order.map(provider => lastKnown.get(provider)).filter((d): d is ProviderUsageDisplay => d !== undefined))
    } catch {
      // A failed poll must not crash the badge; keep last known state.
    } finally {
      inflightRef.current = false
    }
  }, [rpc])

  useEffect(() => {
    mountedRef.current = true
    void refresh()
    const timer = setInterval(() => { void refresh() }, USAGE_POLL_INTERVAL_MS)
    return () => {
      mountedRef.current = false
      clearInterval(timer)
    }
  }, [refresh])

  useEffect(() => {
    if (currentRef.current === undefined) return
    let cancelled = false
    let inflight = false
    const reload = (): void => {
      const read = currentRef.current
      if (read === undefined || inflight) return
      inflight = true
      void read().then(
        (provider) => { if (!cancelled) setCurrent(provider) },
        () => { /* keep the last known provider; the next tick retries */ },
      ).finally(() => { inflight = false })
    }
    reload()
    const timer = setInterval(reload, MODEL_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  const pos = useAnchoredPosition({ open, anchorRef: rootRef, panelRef, side: 'top', gap: PANEL_GAP, margin: PANEL_MARGIN })
  useDismissOnOutsidePointer(rootRef, open, setOpen, panelRef)
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => { document.removeEventListener('keydown', onKeyDown) }
  }, [open])

  if (displays.length === 0) return null

  const collapsed = collapsedDisplays(displays, current)
  const label = collapsed.map(compactSegment).join(' | ')
  const expanded = expandedDisplays(displays, current)
  const title = translate('usageBadgeTitle')

  const toggle = (): void => {
    const next = !open
    setOpen(next)
    if (next) void refresh()
  }

  return (
    <span ref={rootRef} style={styles.anchor}>
      <button
        type="button"
        style={{ ...styles.pill, ...(hover || open ? styles.pillActive : {}) }}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${title} · ${label}`}
        title={title}
        onMouseEnter={() => { setHover(true) }}
        onMouseLeave={() => { setHover(false) }}
        onClick={toggle}
      >
        <IconDataOutline16 />
        <span style={styles.label}>{label}</span>
      </button>
      {open && createPortal(
        <div
          ref={panelRef}
          role="dialog"
          aria-label={title}
          style={{ ...styles.panel, ...(pos ?? MEASURE_STYLE) }}
        >
          <div style={styles.title}>
            <span style={styles.titleLabel}>
              <IconDataOutline16 />
              {title}
            </span>
          </div>
          <div style={styles.titleRule} aria-hidden />
          {expanded.map((d, index) => (
            <section key={d.provider} style={index === 0 ? undefined : styles.section}>
              <div style={styles.providerRow}>
                <span style={styles.providerName}>
                  {d.name}
                  {d.provider === current && <span style={styles.currentTag}>{translate('usageBadgeCurrent')}</span>}
                </span>
                <span style={styles.providerMeta}>
                  {[d.account, d.plan === undefined ? undefined : translate('usagePlan', { plan: d.plan })]
                    .filter((part): part is string => part !== undefined && part !== '')
                    .join(' · ')}
                </span>
              </div>
              <dl style={styles.details}>
                {d.windows.map((w, i) => (
                  <WindowRow key={i} label={usageWindowLabel(translate, w)} window={w} />
                ))}
              </dl>
            </section>
          ))}
        </div>,
        document.body,
      )}
    </span>
  )
}

/** One `dt`/`dd` pair: window name → `25% · 6d1h`, with the bar underneath. */
function WindowRow({ label, window: w }: { label: string; window: UsageWindow }) {
  const percent = usedPercent(w)
  return (
    <>
      <dt style={styles.dt}>{label}</dt>
      <dd style={styles.dd}>
        {percent}%
        {w.resetsAt !== undefined && <span style={styles.reset}> · {windowLabel(w)}</span>}
      </dd>
      <div style={styles.bar} aria-hidden>
        <div style={{ ...styles.barFill, width: `${percent}%`, background: usageBarColor(percent) }} />
      </div>
    </>
  )
}

/**
 * Unplaced portal panel: hidden but laid out so the clamp measures real
 * dimensions (the `useAnchoredPosition` measure pass).
 */
const MEASURE_STYLE: CSSProperties = { visibility: 'hidden', left: 0, top: 0 }

const styles: Record<string, CSSProperties> = {
  anchor: { minWidth: 0, display: 'inline-flex' },
  // Mirrors the host StatsPills pill so the badge reads as a sibling of the
  // shipped time/token pills.
  pill: {
    boxSizing: 'border-box', maxWidth: '100%',
    color: 'var(--dsw-alias-label-tertiary)',
    font: 'inherit', fontSize: 'var(--dsh-content-font-size-secondary, 13px)',
    fontVariantNumeric: 'tabular-nums', lineHeight: '20px', whiteSpace: 'nowrap',
    background: 'transparent', border: 'none', borderRadius: 24,
    alignItems: 'center', gap: 6, padding: '1px 8px', display: 'inline-flex', cursor: 'pointer',
  },
  pillActive: {
    background: 'var(--dsw-alias-interactive-bg-hover)',
    color: 'var(--dsw-alias-label-secondary)',
  },
  label: { textOverflow: 'ellipsis', minWidth: 0, overflow: 'hidden' },
  // Mirrors the host stat-dialog panel.
  panel: {
    position: 'fixed', zIndex: 1100, boxSizing: 'border-box',
    background: 'var(--dsw-specific-menu)',
    width: 'max-content', minWidth: 'min(300px, 100vw - 24px)', maxWidth: 'min(440px, 100vw - 24px)',
    boxShadow: 'var(--dsw-elevation-prominent)',
    color: 'var(--dsw-alias-label-secondary)', cursor: 'default',
    border: 0, borderRadius: 12, padding: 16, fontSize: 12, lineHeight: '18px',
  },
  title: {
    color: 'var(--dsw-alias-label-primary)', display: 'flex',
    justifyContent: 'space-between', gap: 16, marginBottom: 8, fontWeight: 500,
  },
  titleLabel: { alignItems: 'center', gap: 6, minWidth: 0, display: 'inline-flex' },
  titleRule: { borderTop: '0.5px solid var(--dsw-alias-border-l2)', marginBottom: 10 },
  section: { marginTop: 12, paddingTop: 10, borderTop: '0.5px solid var(--dsw-alias-border-l2)' },
  providerRow: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16, marginBottom: 6,
  },
  providerName: { color: 'var(--dsw-alias-label-primary)', fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 6 },
  currentTag: {
    fontSize: 10, lineHeight: '14px', fontWeight: 400, padding: '0 5px', borderRadius: 7,
    color: 'var(--dsw-alias-label-secondary)', background: 'var(--dsw-alias-interactive-bg-hover)',
  },
  providerMeta: {
    color: 'var(--dsw-alias-label-tertiary)', minWidth: 0, overflow: 'hidden',
    textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  details: {
    color: 'var(--dsw-alias-label-tertiary)', display: 'grid',
    gridTemplateColumns: 'minmax(76px, auto) minmax(0, 1fr)', gap: '4px 16px', margin: 0,
  },
  dt: { minWidth: 0, margin: 0 },
  dd: {
    minWidth: 0, margin: 0, color: 'var(--dsw-alias-label-secondary)',
    fontVariantNumeric: 'tabular-nums', textAlign: 'right',
  },
  reset: { color: 'var(--dsw-alias-label-tertiary)' },
  bar: {
    gridColumn: '1 / -1', height: 4, borderRadius: 2, overflow: 'hidden',
    background: 'var(--dsw-alias-border-l2)', marginBottom: 2,
  },
  barFill: { height: '100%', borderRadius: 2 },
}
