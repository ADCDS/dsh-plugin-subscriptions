/**
 * Translate between the harness message vocabulary and the Anthropic Messages
 * API wire format used by the claude provider: request message assembly, tool
 * schema mapping, and a push-model SSE-event → StreamChunk state machine
 * ({@link AnthropicStreamTranslator}) so tests need no streams.
 */

import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  LlmError,
} from '@deepseek-ai/dsh-llm'
import { ToolCallId } from '../compat.js'
import type {
  ContentBlock,
  StreamChunk,
  TokenUsage,
  ToolSchema,
} from '@deepseek-ai/dsh-llm'
import { parseSse } from './sse.js'
import type { ResolvedToolResultBlock, TranslatableMessage } from './resolved.js'

/**
 * The Claude Code identity block. The subscription endpoint rejects requests
 * that do not present as Claude Code, so this block is REQUIRED as the first
 * system entry on every request.
 */
export const CLAUDE_CODE_IDENTITY = 'You are Claude Code, Anthropic\'s official CLI for Claude.'

/** Tags wrapping a mid-conversation system message where it sits in the history. */
export const SYSTEM_REMINDER_OPEN = '<system-reminder>'
export const SYSTEM_REMINDER_CLOSE = '</system-reminder>'

/**
 * How far apart consecutive message breakpoints sit, in content blocks.
 *
 * A breakpoint looks back at most 20 blocks for an entry an earlier request
 * wrote, so marks must stay closer than that: one agentic turn can append a
 * dozen tool_use/tool_result blocks at once, and a single trailing mark would
 * silently fall out of range and rebuild the whole prefix.
 */
export const CACHE_BLOCK_STRIDE = 15

/**
 * Message breakpoints per request. Anthropic allows four in total and the
 * last `system` block takes the fourth, so three are left for the history —
 * enough to tolerate a turn appending roughly {@link CACHE_BLOCK_STRIDE} × 3
 * blocks before a read is lost.
 */
export const MESSAGE_CACHE_BREAKPOINTS = 3

/**
 * How long Anthropic keeps a cache entry after its last use: five minutes
 * (the API default) or one hour. A one-hour entry is written at twice the base
 * input price instead of 1.25×, and reads cost the same either way.
 */
export type PromptCacheTtl = '5m' | '1h'

/** The API default; a marker at this TTL carries no `ttl` field at all. */
export const DEFAULT_PROMPT_CACHE_TTL: PromptCacheTtl = '5m'

/**
 * Build one `cache_control` marker.
 *
 * Five minutes is spelled by omitting `ttl`, so a request that does not opt in
 * stays byte-identical to what earlier releases sent.
 * @param ttl - cache lifetime; defaults to five minutes.
 * @returns a fresh marker object (callers attach it to one block each).
 */
export function cacheControl(ttl: PromptCacheTtl = DEFAULT_PROMPT_CACHE_TTL): Record<string, unknown> {
  return ttl === '1h' ? { type: 'ephemeral', ttl: '1h' } : { type: 'ephemeral' }
}

/** One Anthropic request message. */
export interface AnthropicMessage {
  role: 'user' | 'assistant'
  content: Record<string, unknown>[]
}

/** Preserve native image blocks, retaining the existing text-only wire shape. */
function toolResultContent(block: ResolvedToolResultBlock): string | Record<string, unknown>[] {
  if (!block.content.some(part => part.type === 'image' && 'dataBase64' in part)) {
    return block.content.map(part => (part.type === 'text' ? part.text : '')).join('')
  }
  const content: Record<string, unknown>[] = []
  for (const part of block.content) {
    if (part.type === 'text' && part.text.length > 0) content.push({ type: 'text', text: part.text })
    if (part.type === 'image' && 'dataBase64' in part) {
      content.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.dataBase64 } })
    }
  }
  return content
}

/** Parse a tool call's raw JSON arguments into Anthropic's object-shaped `input`. */
function parseToolInput(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return {}
  } catch {
    // The model produced malformed JSON; an empty object keeps the request valid.
    return {}
  }
}

/**
 * Move a user message's `tool_result` blocks into one contiguous run at the
 * front, preserving the relative order of both groups.
 *
 * Anthropic answers every `tool_use` against the blocks that *lead* the next
 * message, so a block of any other kind before or between the results reads
 * as a call left unanswered and the request is rejected. The harness merges
 * everything queued for one user turn into a single message, and a parallel
 * tool batch arrives as one result message per call, so any context spliced
 * mid-batch lands between two results. Restoring the run here keeps that
 * independent of delivery order. Order *among* the results does not matter.
 * @param message - one assembled user message, reordered in place.
 */
function leadWithToolResults(message: AnthropicMessage): void {
  const firstOther = message.content.findIndex(block => block.type !== 'tool_result')
  if (firstOther === -1) return
  if (!message.content.slice(firstOther).some(block => block.type === 'tool_result')) return
  message.content = [
    ...message.content.filter(block => block.type === 'tool_result'),
    ...message.content.filter(block => block.type !== 'tool_result'),
  ]
}

/**
 * Index of the first non-system message; `messages.length` when every message
 * is a system one.
 *
 * A system message before the conversation starts is the operator's opening
 * instruction and belongs in the `system` slot. One that arrives later is
 * mid-conversation context, and hoisting it into `system` would move bytes in
 * front of the whole history — invalidating every cached turn behind it — so
 * it stays where it is, as a reminder block in `messages`.
 * @param messages - ordered conversation messages.
 * @returns the boundary index separating the two.
 */
function conversationStart(messages: readonly TranslatableMessage[]): number {
  const index = messages.findIndex(message => message.role !== 'system')
  return index === -1 ? messages.length : index
}

/**
 * Convert harness messages into Anthropic messages. Consecutive same-role
 * messages merge into one message with multiple content blocks; tool results
 * arrive as user messages with `tool_result` blocks, which a merged user
 * message keeps in one leading run ({@link leadWithToolResults}); system-role
 * messages before the conversation starts are handled by
 * {@link toAnthropicSystem} and skipped here, while a later one rides in
 * place as a user-role `<system-reminder>` block.
 * Reasoning blocks are not replayed (v1). Images must arrive pre-resolved
 * ({@link TranslatableMessage}); an unresolved ImageBlock is skipped because
 * its bytes are unreachable here.
 * @param messages - ordered conversation messages with resolved images.
 * @returns Anthropic messages in conversation order.
 */
export function toAnthropicMessages(messages: readonly TranslatableMessage[]): AnthropicMessage[] {
  const out: AnthropicMessage[] = []
  const start = conversationStart(messages)
  for (const [index, message] of messages.entries()) {
    // A leading system message is an opening instruction; toAnthropicSystem
    // owns those. A later one rides here so the cached prefix ahead of it
    // stays byte-identical.
    if (message.role === 'system' && index < start) continue
    const role = message.role === 'assistant' ? 'assistant' : 'user'
    const blocks: Record<string, unknown>[] = []
    if (message.role === 'tool') {
      const id = message.toolCallId ?? message.tool_call_id
        ?? (message.source?.kind === 'tool' ? String(message.source.callId) : undefined)
      if (id === undefined) throw new LlmError('tool result has no call id', 'INVALID_REQUEST')
      blocks.push({
        type: 'tool_result',
        tool_use_id: id,
        content: toolResultContent({ type: 'tool-result', toolCallId: ToolCallId(id), content: message.content }),
        ...message.isError === true ? { is_error: true } : {},
      })
      const last = out[out.length - 1]
      if (last?.role === 'user') last.content.push(...blocks)
      else out.push({ role: 'user', content: blocks })
      continue
    }
    for (const block of message.content) {
      switch (block.type) {
        case 'text':
          blocks.push({
            type: 'text',
            text: message.role === 'system'
              ? `${SYSTEM_REMINDER_OPEN}${block.text}${SYSTEM_REMINDER_CLOSE}`
              : block.text,
          })
          break
        case 'tool-call':
          // Anthropic accepts `tool_use` only in assistant messages, and only
          // when a matching `tool_result` follows. A tool call in any other
          // role is replayed narrative — a settled subagent's closing message
          // spliced into the parent as a user-role notice carries the calls it
          // died holding, which no result will ever answer — so it rides as
          // descriptive text instead of a call the API would reject.
          blocks.push(role === 'assistant'
            ? {
                type: 'tool_use',
                id: String(block.id),
                name: block.name,
                input: parseToolInput(block.arguments),
              }
            : { type: 'text', text: `[tool call ${block.name}: ${block.arguments}]` })
          break
        case 'tool-result':
          blocks.push({
            type: 'tool_result',
            tool_use_id: String(block.toolCallId),
            content: toolResultContent(block),
            ...block.isError === true ? { is_error: true } : {},
          })
          break
        case 'image':
          if ('dataBase64' in block) {
            blocks.push({
              type: 'image',
              source: { type: 'base64', media_type: block.mediaType, data: block.dataBase64 },
            })
          }
          // An unresolved ImageBlock carries only an attachment reference; the
          // adapter resolves images before translation, so this is skipped.
          break
        default:
          // reasoning (not replayed), unknown blocks.
          break
      }
    }
    if (blocks.length === 0) continue
    const last = out[out.length - 1]
    if (last !== undefined && last.role === role) last.content.push(...blocks)
    else out.push({ role, content: blocks })
  }
  for (const message of out) {
    if (message.role === 'user') leadWithToolResults(message)
  }
  return out
}

/**
 * Mark the conversation's cache breakpoints in place: the last content block,
 * then one every {@link CACHE_BLOCK_STRIDE} blocks backwards, {@link
 * MESSAGE_CACHE_BREAKPOINTS} in total.
 *
 * The history is append-only, so the block one request marks last is
 * byte-identical in the next — that entry is what the next request reads.
 * Marks are counted across the flattened block sequence, not per message,
 * because the lookback window Anthropic walks counts blocks the same way.
 * @param messages - assembled Anthropic messages, marked in place.
 * @param ttl - cache lifetime for every mark; defaults to five minutes.
 */
export function markMessageCache(messages: readonly AnthropicMessage[], ttl?: PromptCacheTtl): void {
  const blocks = messages.flatMap(message => message.content)
  for (let mark = 0; mark < MESSAGE_CACHE_BREAKPOINTS; mark++) {
    const at = blocks.length - 1 - mark * CACHE_BLOCK_STRIDE
    if (at < 0) return
    blocks[at].cache_control = cacheControl(ttl)
  }
}

/**
 * Build the Anthropic `system` array: the mandatory Claude Code identity
 * block, then the explicit system prompt, then any system-role messages.
 * @param system - explicit system prompt, when set.
 * @param messages - conversation messages; the system-role text preceding the
 * conversation is appended, and a later one is left to {@link toAnthropicMessages}.
 * @param ttl - cache lifetime for the tools+system mark; defaults to five minutes.
 * @returns the system content blocks.
 */
export function toAnthropicSystem(
  system?: string,
  messages?: readonly TranslatableMessage[],
  ttl?: PromptCacheTtl,
): Record<string, unknown>[] {
  const blocks: Record<string, unknown>[] = [{ type: 'text', text: CLAUDE_CODE_IDENTITY }]
  if (system !== undefined && system.length > 0) blocks.push({ type: 'text', text: system })
  const history = messages ?? []
  for (const message of history.slice(0, conversationStart(history))) {
    for (const block of message.content) {
      if (block.type === 'text') blocks.push({ type: 'text', text: block.text })
    }
  }
  // `tools` renders ahead of `system`, so this one marker caches both. It is
  // deliberately separate from the message marks: a tool_choice or thinking
  // change invalidates the messages tier only, and this entry survives it.
  //
  // Anthropic processes `tools`, `system`, then `messages`, and rejects a
  // one-hour mark that follows a five-minute one (HTTP 400). This mark and the
  // message marks therefore share one TTL, which can never hit that ordering.
  blocks[blocks.length - 1].cache_control = cacheControl(ttl)
  return blocks
}

/**
 * Map harness tool schemas to Anthropic tools, in name order.
 *
 * `tools` renders at position 0 of the cached prefix, so any reordering
 * invalidates every cache entry behind it — `system` and the whole
 * conversation included. Registration order belongs to the caller and plugin
 * load order can differ between processes, so the wire order is fixed here
 * instead. Anthropic selects a tool by name; the array order carries nothing.
 * @param tools - tool schemas from the request.
 * @returns Anthropic `tools` array entries, ordered by tool name.
 */
export function toAnthropicTools(tools: readonly ToolSchema[]): Record<string, unknown>[] {
  return [...tools]
    .sort((left, right) => (left.name < right.name ? -1 : left.name > right.name ? 1 : 0))
    .map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }))
}

/** The subset of Anthropic SSE event shapes this translator reads. */
export interface AnthropicStreamEvent {
  type: string
  index?: number
  message?: {
    id?: string
    /** The model serving this message: a fallback model when a refusal was rerouted before any output. */
    model?: string
    usage?: {
      input_tokens?: number
      output_tokens?: number
      cache_read_input_tokens?: number
      cache_creation_input_tokens?: number
    }
  }
  content_block?: {
    type?: string
    id?: string
    name?: string
    /** `fallback` blocks: the declining model, the model that continues, and why. */
    from?: { model?: string }
    to?: { model?: string }
    trigger?: { type?: string; category?: string | null }
  }
  delta?: {
    type?: string
    text?: string
    thinking?: string
    partial_json?: string
    stop_reason?: string
    /** Set with `stop_reason: refusal`: the classifier's policy area and explanation. */
    stop_details?: {
      type?: string
      category?: string | null
      explanation?: string | null
      recommended_model?: string
    } | null
  }
  usage?: { output_tokens?: number }
  error?: { type?: string; message?: string }
}

/** One open harness block under assembly. */
interface OpenBlock {
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  callId: string
  name?: string
}

/** Assemble the final ContentBlock for one open block. */
function closeBlock(block: OpenBlock): ContentBlock {
  switch (block.kind) {
    case 'text':
      return { type: 'text', text: block.text }
    case 'reasoning':
      return { type: 'reasoning', text: block.text }
    case 'tool-call':
      return {
        type: 'tool-call',
        id: ToolCallId(block.callId),
        name: block.name ?? '',
        arguments: block.text,
      }
  }
}

/**
 * Classify an Anthropic `error` event into a thrown LlmError.
 * @param error - the wire error object.
 * @returns the mapped error.
 */
export function anthropicFailure(error: { type?: string; message?: string } | undefined): LlmError {
  const type = error?.type ?? 'unknown_error'
  const message = error?.message ?? `Anthropic reported ${type}`
  if (type === 'invalid_request_error' && /prompt is too long/i.test(message)) {
    return new LlmError(message, CONTEXT_WINDOW_EXCEEDED_CODE)
  }
  if (type === 'rate_limit_error') return new LlmError(message, 'RATE_LIMIT')
  if (type === 'authentication_error') return new LlmError(message, 'AUTH')
  return new LlmError(message, 'SERVER')
}

/** Failure code of a safety-classifier decline that no fallback answered; not retried, and no account is cooled. */
export const REFUSAL_CODE = 'REFUSAL'

/** Per-request options of one Anthropic stream. */
export interface AnthropicStreamOptions {
  /**
   * The request carried server-side `fallbacks`: hold tool calls until the
   * message ends, because a classifier handoff discards the declined model's.
   */
  holdToolCalls?: boolean
  /** The requested model id, to tell when another model served the reply. */
  requestedModel?: string
}

/**
 * Which model answered when it was not the requested one. Carried on the
 * successful `finish` chunk as `servedBy`; hosts that do not read it ignore it.
 */
export interface ServedBy {
  model: string
  reason: 'refusal-fallback'
  from?: string
  category?: string
}

/**
 * Push-model Anthropic SSE translator: feed each parsed event object to
 * {@link push} and collect the emitted harness StreamChunks. Block indexes
 * are allocated in first-seen order; `usage` is emitted before the terminal
 * `finish`, and nothing is emitted after it. `error` events throw
 * {@link LlmError}.
 */
export class AnthropicStreamTranslator {
  constructor(private readonly streamOptions: AnthropicStreamOptions = {}) {}

  private blocks = new Map<number, OpenBlock>()
  /** Tool calls held until `message_stop`, by wire index, while a classifier handoff could still discard them. */
  private held = new Map<number, OpenBlock>()
  /** The model `message_start` named. */
  private startModel: string | undefined
  /** The handoff a `fallback` block announced. */
  private fallback: { from?: string; to?: string; category?: string } | undefined
  /** The classifier decline a `refusal` stop carried. */
  private refusal: { category?: string; explanation?: string; recommended?: string } | undefined
  private nextIndex = 0
  private sawAnyBlock = false
  private pendingUsage: { inputTokens: number; cacheReadTokens?: number; cacheWriteTokens?: number } | undefined
  private outputTokens: number | undefined
  private stopReason: 'stop' | 'tool-calls' | 'max-tokens' = 'stop'
  /** The wire `stop_reason`, kept verbatim so an empty completion can say why it stopped. */
  private wireStopReason: string | undefined
  /** Set by `model_context_window_exceeded`: the input plus output cap overran the window. */
  private contextExceeded = false
  /** Content block types this translator does not render, named when a reply comes back empty. */
  private skippedBlockTypes = new Set<string>()
  private messageId: string | undefined
  private usageEmitted = false
  /** Set once `message_stop` produced the terminal finish chunk. */
  terminated = false

  private open(wireIndex: number, kind: OpenBlock['kind'], chunks: StreamChunk[], callId = '', name?: string): OpenBlock {
    const block: OpenBlock = {
      index: this.nextIndex++,
      kind,
      text: '',
      callId,
      ...name === undefined ? {} : { name },
    }
    this.blocks.set(wireIndex, block)
    this.sawAnyBlock = true
    chunks.push({ type: 'block-start', index: block.index, blockType: kind })
    return block
  }

  /**
   * The failure text for a completed message with no content block. It names
   * the wire stop reason (an unrecognized one such as `refusal` lands here as
   * well), the output tokens the message reported, and its id, so an empty
   * reply can be told apart from a refusal or a stream that never stopped.
   */
  private emptyResponseMessage(): string {
    const facts = this.responseFacts()
    if (this.skippedBlockTypes.size > 0) facts.push(`unrendered blocks: ${[...this.skippedBlockTypes].join(', ')}`)
    return `model returned a completed response with no content (${facts.join(', ')})`
  }

  /** The failure text for a classifier decline that no fallback answered. */
  private refusalMessage(): string {
    const refusal = this.refusal ?? {}
    const area = refusal.category ?? 'no named category'
    const why = refusal.explanation === undefined ? '' : `: ${refusal.explanation}`
    const retry = refusal.recommended === undefined ? '' : `; Anthropic recommends retrying on ${refusal.recommended}`
    return `Claude declined the request (${area}${why})${retry} (${this.responseFacts().join(', ')})`
  }

  /**
   * The model that actually answered, when it is not the requested one: a
   * `fallback` block's target, or the model `message_start` named when sticky
   * routing sent the turn straight to an earlier fallback.
   */
  private servedBy(): ServedBy | undefined {
    if (this.fallback?.to !== undefined) {
      return {
        model: this.fallback.to,
        reason: 'refusal-fallback',
        ...this.fallback.from === undefined ? {} : { from: this.fallback.from },
        ...this.fallback.category === undefined ? {} : { category: this.fallback.category },
      }
    }
    const requested = this.streamOptions.requestedModel
    const started = this.startModel
    if (requested === undefined || started === undefined || started === requested || started.startsWith(`${requested}-`)) {
      return undefined
    }
    return { model: started, reason: 'refusal-fallback', from: requested }
  }

  /** The stop reason, output tokens and message id a terminal failure names. */
  private responseFacts(): string[] {
    return [
      `stop_reason ${this.wireStopReason ?? 'not sent'}`,
      `${String(this.outputTokens ?? 0)} output tokens`,
      ...this.messageId === undefined ? [] : [this.messageId],
    ]
  }

  private emitUsage(chunks: StreamChunk[]): void {
    if (this.usageEmitted) return
    this.usageEmitted = true
    const usage: TokenUsage = {
      inputTokens: this.pendingUsage?.inputTokens ?? 0,
      outputTokens: this.outputTokens ?? 0,
      ...this.pendingUsage?.cacheReadTokens !== undefined
        ? { cacheReadTokens: this.pendingUsage.cacheReadTokens }
        : {},
      ...this.pendingUsage?.cacheWriteTokens !== undefined
        ? { cacheWriteTokens: this.pendingUsage.cacheWriteTokens }
        : {},
    }
    chunks.push({ type: 'usage', usage })
  }

  /**
   * Process one parsed Anthropic SSE event.
   * @param event - the parsed event object.
   * @returns the StreamChunks this event produced (possibly none).
   */
  push(event: AnthropicStreamEvent): StreamChunk[] {
    if (this.terminated) return []
    const chunks: StreamChunk[] = []
    switch (event.type) {
      case 'message_start': {
        if (typeof event.message?.id === 'string' && event.message.id.length > 0) this.messageId = event.message.id
        if (typeof event.message?.model === 'string' && event.message.model.length > 0) this.startModel = event.message.model
        const usage = event.message?.usage
        if (usage !== undefined) {
          this.pendingUsage = {
            inputTokens: usage.input_tokens ?? 0,
            ...usage.cache_read_input_tokens !== undefined
              ? { cacheReadTokens: usage.cache_read_input_tokens }
              : {},
            ...usage.cache_creation_input_tokens !== undefined
              ? { cacheWriteTokens: usage.cache_creation_input_tokens }
              : {},
          }
          this.outputTokens = usage.output_tokens ?? this.outputTokens
        }
        return chunks
      }
      case 'content_block_start': {
        const wireIndex = event.index ?? 0
        const block = event.content_block
        switch (block?.type) {
          case 'text':
            this.open(wireIndex, 'text', chunks)
            break
          case 'thinking':
            this.open(wireIndex, 'reasoning', chunks)
            break
          case 'tool_use': {
            if (this.streamOptions.holdToolCalls === true) {
              this.held.set(wireIndex, {
                index: -1, kind: 'tool-call', text: '', callId: block.id ?? '',
                ...block.name === undefined ? {} : { name: block.name },
              })
              break
            }
            const opened = this.open(wireIndex, 'tool-call', chunks, block.id ?? '', block.name)
            chunks.push({
              type: 'tool-call-delta',
              index: opened.index,
              id: ToolCallId(opened.callId),
              ...block.name === undefined ? {} : { name: block.name },
              argumentsDelta: '',
            })
            break
          }
          case 'fallback':
            // A classifier declined and another model continues. The declined
            // model's tool calls must not run; its text stays as the context the
            // fallback model continued from.
            this.fallback = {
              ...typeof block.from?.model === 'string' ? { from: block.from.model } : {},
              ...typeof block.to?.model === 'string' ? { to: block.to.model } : {},
              ...typeof block.trigger?.category === 'string' ? { category: block.trigger.category } : {},
            }
            this.held.clear()
            break
          default:
            if (typeof block?.type === 'string') this.skippedBlockTypes.add(block.type)
            break
        }
        return chunks
      }
      case 'content_block_delta': {
        const wireIndex = event.index ?? 0
        const held = this.held.get(wireIndex)
        if (held !== undefined) {
          if (event.delta?.type === 'input_json_delta') held.text += event.delta.partial_json ?? ''
          return chunks
        }
        const block = this.blocks.get(wireIndex)
        const delta = event.delta
        if (block === undefined || delta === undefined) return chunks
        switch (delta.type) {
          case 'text_delta':
            block.text += delta.text ?? ''
            chunks.push({ type: 'text-delta', index: block.index, text: delta.text ?? '' })
            break
          case 'thinking_delta':
            block.text += delta.thinking ?? ''
            chunks.push({ type: 'reasoning-delta', index: block.index, text: delta.thinking ?? '' })
            break
          case 'input_json_delta':
            block.text += delta.partial_json ?? ''
            chunks.push({
              type: 'tool-call-delta',
              index: block.index,
              id: ToolCallId(block.callId),
              ...block.name === undefined ? {} : { name: block.name },
              argumentsDelta: delta.partial_json ?? '',
            })
            break
          default:
            // signature_delta and future deltas carry no harness content.
            break
        }
        return chunks
      }
      case 'content_block_stop': {
        const wireIndex = event.index ?? 0
        const block = this.blocks.get(wireIndex)
        if (block === undefined) return chunks
        this.blocks.delete(wireIndex)
        chunks.push({ type: 'block-end', index: block.index, block: closeBlock(block) })
        return chunks
      }
      case 'message_delta': {
        if (event.usage?.output_tokens !== undefined) this.outputTokens = event.usage.output_tokens
        if (typeof event.delta?.stop_reason === 'string') this.wireStopReason = event.delta.stop_reason
        if (event.delta?.stop_reason === 'refusal') {
          const details = event.delta.stop_details
          this.refusal = {
            ...typeof details?.category === 'string' ? { category: details.category } : {},
            ...typeof details?.explanation === 'string' && details.explanation.length > 0 ? { explanation: details.explanation } : {},
            ...typeof details?.recommended_model === 'string' ? { recommended: details.recommended_model } : {},
          }
        }
        switch (event.delta?.stop_reason) {
          case 'end_turn':
          case 'stop_sequence':
            this.stopReason = 'stop'
            break
          case 'tool_use':
            this.stopReason = 'tool-calls'
            break
          case 'max_tokens':
            this.stopReason = 'max-tokens'
            break
          case 'model_context_window_exceeded':
            // The output was cut at the window: partial content reads like a length stop.
            this.stopReason = 'max-tokens'
            this.contextExceeded = true
            break
          default:
            break
        }
        return chunks
      }
      case 'message_stop': {
        this.terminated = true
        for (const [wireIndex, block] of [...this.blocks]) {
          this.blocks.delete(wireIndex)
          chunks.push({ type: 'block-end', index: block.index, block: closeBlock(block) })
        }
        // A standing refusal discards the declined model's partial tool work;
        // otherwise no handoff can take them any more, so release them.
        if (this.wireStopReason !== 'refusal') {
          for (const held of this.held.values()) {
            const opened = this.open(-1, 'tool-call', chunks, held.callId, held.name)
            this.blocks.delete(-1)
            opened.text = held.text
            chunks.push({
              type: 'tool-call-delta',
              index: opened.index,
              id: ToolCallId(opened.callId),
              ...opened.name === undefined ? {} : { name: opened.name },
              argumentsDelta: held.text,
            })
            chunks.push({ type: 'block-end', index: opened.index, block: closeBlock(opened) })
          }
        }
        this.held.clear()
        this.emitUsage(chunks)
        if (this.wireStopReason === 'refusal') {
          // The classifier's decline stands (no fallback, or every fallback declined).
          // Partial output is incomplete, so the turn fails rather than ending on it.
          chunks.push({
            type: 'finish',
            reason: { kind: 'error', failure: { message: this.refusalMessage(), code: REFUSAL_CODE } },
          })
        } else if (this.contextExceeded && !this.sawAnyBlock) {
          // Nothing fit: report the overflow, which the host answers by compacting and retrying.
          chunks.push({
            type: 'finish',
            reason: {
              kind: 'error',
              failure: {
                message: `the conversation filled the model's context window before it produced any output (${this.responseFacts().join(', ')})`,
                code: CONTEXT_WINDOW_EXCEEDED_CODE,
              },
            },
          })
        } else if (this.stopReason === 'stop' && !this.sawAnyBlock) {
          chunks.push({
            type: 'finish',
            reason: {
              kind: 'error',
              failure: { message: this.emptyResponseMessage(), code: EMPTY_RESPONSE_CODE },
            },
          })
        } else {
          const finish: StreamChunk = { type: 'finish', reason: { kind: this.stopReason } }
          const servedBy = this.servedBy()
          chunks.push(servedBy === undefined ? finish : { ...finish, servedBy } as StreamChunk)
        }
        return chunks
      }
      case 'error':
        throw anthropicFailure(event.error)
      default:
        // ping and future event types carry no harness content.
        return chunks
    }
  }
}

/**
 * Consume an Anthropic SSE byte stream and yield harness StreamChunks.
 * @param stream - raw response body.
 * @param onActivity - transport-activity callback for the idle watchdog.
 * @param options - per-request translation options (server-side fallback).
 * @returns the chunk stream; throws when the stream ends before `message_stop`.
 */
export async function* streamAnthropic(
  stream: ReadableStream<Uint8Array>,
  onActivity?: () => void,
  options: AnthropicStreamOptions = {},
): AsyncGenerator<StreamChunk> {
  const translator = new AnthropicStreamTranslator(options)
  for await (const sseEvent of parseSse(stream, onActivity)) {
    let event: AnthropicStreamEvent
    try {
      event = JSON.parse(sseEvent.data) as AnthropicStreamEvent
    } catch {
      throw new LlmError(`malformed SSE payload: ${sseEvent.data.slice(0, 120)}`, 'MALFORMED_RESPONSE')
    }
    yield* translator.push(event)
    if (translator.terminated) return
  }
  throw new LlmError('Anthropic SSE stream ended before message_stop', 'STREAM_CLOSED')
}
