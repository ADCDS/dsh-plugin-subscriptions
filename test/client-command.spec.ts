import { test } from 'node:test'
import assert from 'node:assert/strict'
import { fastCommandDescription } from '../src/client/fast-command.js'

test('/fast description is a callable locale resolver', () => {
  let locale = 'commandFast'
  const description = fastCommandDescription(() => locale)
  assert.equal(description(), 'commandFast')
  locale = '切换速度档'
  assert.equal(description(), '切换速度档')
})
