import { test } from 'node:test'
import assert from 'node:assert/strict'
import { getDefaultAutoSelectFamilyAttemptTimeout, setDefaultAutoSelectFamilyAttemptTimeout } from 'node:net'
import { ensureConnectAttemptTimeout, MIN_CONNECT_ATTEMPT_TIMEOUT_MS, restoreConnectAttemptTimeout } from '../src/http.js'

test('ensureConnectAttemptTimeout raises a short Happy Eyeballs budget and never lowers a longer one', () => {
  const original = getDefaultAutoSelectFamilyAttemptTimeout()
  try {
    setDefaultAutoSelectFamilyAttemptTimeout(250)
    assert.equal(ensureConnectAttemptTimeout(), 250)
    assert.equal(getDefaultAutoSelectFamilyAttemptTimeout(), MIN_CONNECT_ATTEMPT_TIMEOUT_MS)
    setDefaultAutoSelectFamilyAttemptTimeout(MIN_CONNECT_ATTEMPT_TIMEOUT_MS + 1000)
    assert.equal(ensureConnectAttemptTimeout(), MIN_CONNECT_ATTEMPT_TIMEOUT_MS + 1000)
    assert.equal(getDefaultAutoSelectFamilyAttemptTimeout(), MIN_CONNECT_ATTEMPT_TIMEOUT_MS + 1000)
    restoreConnectAttemptTimeout(250)
    assert.equal(getDefaultAutoSelectFamilyAttemptTimeout(), 250)
  } finally {
    setDefaultAutoSelectFamilyAttemptTimeout(original)
  }
})
