import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import { toAntigravityContents } from '../../src/translate/antigravity.js'

const MODEL = 'gemini-3.8-flash-tiered'

type Messages = Parameters<typeof toAntigravityContents>[0]

/**
 * Build one tool result whose text is exactly the supplied payload, then read
 * back the response the adapter would put on the wire.
 */
function responseFor(text: string, isError = false): unknown {
  const messages = [
    {
      role: 'assistant',
      content: [{ type: 'tool-call', id: 'c1', name: 'tool', arguments: '{}' }],
    },
    {
      role: 'user',
      content: [
        {
          type: 'tool-result',
          toolCallId: 'c1',
          content: [{ type: 'text', text }],
          isError,
        },
      ],
    },
  ] as unknown as Messages
  for (const entry of toAntigravityContents(messages, MODEL)) {
    for (const part of entry.parts) {
      if (part.functionResponse !== undefined) return part.functionResponse.response
    }
  }
  return undefined
}

/** Antigravity maps this field to a singular protobuf Struct. */
function assertStructCompatible(value: unknown): void {
  assert.equal(typeof value, 'object')
  assert.notEqual(value, null)
  assert.equal(Array.isArray(value), false)
}

test('a JSON object tool result is preserved verbatim', () => {
  assert.deepEqual(responseFor('{"a":1}'), { a: 1 })
  assert.deepEqual(responseFor('{"a":{"b":[1,2]}}'), { a: { b: [1, 2] } })
})

test('an array tool result is wrapped in an object', () => {
  assertStructCompatible(responseFor('[{"a":1},{"b":2}]'))
  assert.deepEqual(responseFor('[{"a":1},{"b":2}]'), { output: [{ a: 1 }, { b: 2 }] })
  assert.deepEqual(responseFor('["a","b"]'), { output: ['a', 'b'] })
})

test('empty and nested arrays are wrapped', () => {
  assert.deepEqual(responseFor('[]'), { output: [] })
  assert.deepEqual(responseFor('[[1,2]]'), { output: [[1, 2]] })
})

test('scalar JSON tool results are wrapped', () => {
  assert.deepEqual(responseFor('"hello"'), { output: 'hello' })
  assert.deepEqual(responseFor('42'), { output: 42 })
  assert.deepEqual(responseFor('true'), { output: true })
  assert.deepEqual(responseFor('null'), { output: null })
})

test('non-JSON tool results keep their existing envelope', () => {
  assert.deepEqual(responseFor('plain output'), { output: 'plain output' })
  assert.deepEqual(responseFor('boom', true), { output: 'boom', isError: true })
})

test('multiple tool results in one turn stay Struct compatible', () => {
  const messages = [
    {
      role: 'assistant',
      content: [
        { type: 'tool-call', id: 'c1', name: 'first', arguments: '{}' },
        { type: 'tool-call', id: 'c2', name: 'second', arguments: '{}' },
      ],
    },
    {
      role: 'user',
      content: [
        { type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: '{"ok":1}' }] },
        { type: 'tool-result', toolCallId: 'c2', content: [{ type: 'text', text: '[1,2,3]' }] },
      ],
    },
  ] as unknown as Messages
  const responses: unknown[] = []
  for (const entry of toAntigravityContents(messages, MODEL)) {
    for (const part of entry.parts) {
      if (part.functionResponse !== undefined) responses.push(part.functionResponse.response)
    }
  }
  assert.equal(responses.length, 2)
  assert.deepEqual(responses[0], { ok: 1 })
  assert.deepEqual(responses[1], { output: [1, 2, 3] })
})

test('every JSON payload shape yields a Struct compatible response', () => {
  const payloads = ['{"a":1}', '[1]', '[]', '"s"', '7', 'false', 'null', '[[[]]]', 'plain text']
  for (const payload of payloads) assertStructCompatible(responseFor(payload))
})
