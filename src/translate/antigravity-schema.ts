/** Model-specific tool schemas for Antigravity's Gemini and custom-tool bridges. */
import { LlmError } from '@deepseek-ai/dsh-llm'

type Schema = Record<string, unknown>
const CUSTOM_KEYS = new Set(['type', 'description', 'properties', 'required', 'items', 'enum'])

function object(value: unknown): value is Schema {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Resolve local references before removing definitions; reject cycles and remote references. */
function resolve(value: unknown, root: Schema, refs = new Set<string>()): unknown {
  if (Array.isArray(value)) return value.map(item => resolve(item, root, refs))
  if (!object(value)) return value
  if (typeof value.$ref === 'string') {
    const ref = value.$ref
    if (!ref.startsWith('#/') || refs.has(ref)) {
      throw new LlmError(`Antigravity tool schema has an unsupported reference: ${ref}`, 'INVALID_REQUEST')
    }
    let target: unknown = root
    for (const segment of ref.slice(2).split('/')) {
      const key = segment.replaceAll('~1', '/').replaceAll('~0', '~')
      target = object(target) && Object.hasOwn(target, key) ? target[key] : undefined
    }
    if (!object(target)) throw new LlmError(`Antigravity tool schema reference is missing: ${ref}`, 'INVALID_REQUEST')
    const { $ref: _ref, ...siblings } = value
    return resolve({ ...target, ...siblings }, root, new Set([...refs, ref]))
  }
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['$schema', '$id', '$defs', 'definitions'].includes(key))
    .map(([key, item]) => [key, key === 'properties' && object(item)
      ? Object.fromEntries(Object.entries(item).map(([name, property]) => [name, resolve(property, root, refs)]))
      : resolve(item, root, refs)]))
}

/** Keep property names intact while reducing schema keywords for Claude/GPT-OSS. */
function custom(value: unknown): unknown {
  if (!object(value)) return value
  const schema = { ...value }
  for (const keyword of ['anyOf', 'oneOf']) {
    if (!Array.isArray(schema[keyword])) continue
    const variants = schema[keyword].filter(item => !object(item) || item.type !== 'null')
    if (variants.length !== 1 || !object(variants[0])) {
      throw new LlmError(`Antigravity custom tools cannot represent ${keyword} with multiple alternatives`, 'INVALID_REQUEST')
    }
    Object.assign(schema, variants[0])
  }
  if (schema.allOf !== undefined) {
    throw new LlmError('Antigravity custom tools cannot represent allOf', 'INVALID_REQUEST')
  }
  const out: Schema = {}
  for (const [key, value] of Object.entries(schema)) {
    if (!CUSTOM_KEYS.has(key)) continue
    if (key === 'properties' && object(value)) {
      out[key] = Object.fromEntries(Object.entries(value).map(([name, property]) => [name, custom(property)]))
    } else if (key === 'items') out[key] = custom(value)
    else if (key === 'type' && Array.isArray(value)) {
      const types = value.filter(type => type !== 'null')
      if (types.length !== 1) throw new LlmError('Antigravity custom tool type must have one non-null alternative', 'INVALID_REQUEST')
      out[key] = types[0]
    } else if (key !== 'enum' || Array.isArray(value) && value.every(entry => typeof entry === 'string')) {
      out[key] = value
    }
  }
  return out
}

/** Detach and normalize one tool's root object without changing the registry schema. */
export function antigravityToolParameters(parameters: Schema, legacy: boolean): Schema {
  const resolved = resolve(parameters, parameters) as Schema
  const root = { ...resolved, type: resolved.type ?? 'object' }
  if (root.type !== 'object') throw new LlmError('Antigravity tool parameters must be an object schema', 'INVALID_REQUEST')
  return (legacy ? custom(root) : root) as Schema
}
