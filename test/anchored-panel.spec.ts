import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { placeAbove } from '../src/client/anchored-panel.js'

const viewport = { width: 1000, height: 800 }
const panel = { width: 300, height: 200 }

test('placeAbove sits above the anchor with the gap, aligned to its left edge', () => {
  assert.deepEqual(placeAbove({ left: 100, top: 500 }, panel, viewport, 8, 12), { left: 100, top: 292 })
})

test('placeAbove clamps a panel that would overflow the right edge', () => {
  assert.equal(placeAbove({ left: 900, top: 500 }, panel, viewport, 8, 12).left, 688)
})

test('placeAbove clamps a panel that would overflow the top edge', () => {
  assert.equal(placeAbove({ left: 100, top: 40 }, panel, viewport, 8, 12).top, 12)
})

test('placeAbove keeps the top and left edges visible when the panel cannot fit', () => {
  assert.deepEqual(placeAbove({ left: 50, top: 50 }, { width: 2000, height: 2000 }, viewport, 8, 12), { left: 12, top: 12 })
})

test('the badge does not import the positioning hook DSH removed after 0.2.2-rc.1', () => {
  const source = readFileSync(new URL('../../src/client/SubscriptionUsageBadge.tsx', import.meta.url), 'utf8')
  assert.doesNotMatch(source, /useAnchoredPosition\b(?!`)/)
})
