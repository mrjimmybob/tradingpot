/**
 * Regression tests for the /bots Start/Stop/Pause controls.
 *
 * Bug: the per-row action buttons on the bots list were rendered with no
 * onClick handler, so clicking Start/Pause/Stop did nothing — no API request,
 * no state change — while the same actions worked from the bot detail page.
 *
 * These tests assert the list controls now hit the same backend endpoints the
 * detail page uses (/api/bots/:id/{start,pause,stop}) and that the table
 * refreshes to the new status afterwards without a page reload.
 *
 * Lives under src/ on purpose: the vitest `include` glob only runs src/**.
 */
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import BotList from './BotList'
import { ToastProvider } from '../components/Toast'

interface FakeBot {
  id: number
  name: string
  trading_pair: string
  strategy: string
  status: string
  total_pnl: number
  is_dry_run: boolean
  created_at: string
}

function makeBot(overrides: Partial<FakeBot> = {}): FakeBot {
  return {
    id: 1,
    name: 'TestBot1',
    trading_pair: 'BTC/USDT',
    strategy: 'mean_reversion',
    status: 'created',
    total_pnl: 0,
    is_dry_run: true,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

/** Minimal Response stub for the mocked global fetch. */
function jsonResponse(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  return Promise.resolve({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: () => Promise.resolve(body),
  } as Response)
}

/**
 * Stateful fetch mock: GET /api/bots returns the current bots; a POST to an
 * action endpoint flips the bot's status (mirroring the backend) so a refetch
 * reflects the transition. Records every call for assertions.
 */
function installFetchMock(initial: FakeBot[]) {
  const bots = initial.map((b) => ({ ...b }))
  const calls: { url: string; method: string }[] = []
  const ACTION_STATUS: Record<string, string> = {
    start: 'running',
    pause: 'paused',
    stop: 'stopped',
  }

  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = (init?.method ?? 'GET').toUpperCase()
    calls.push({ url, method })

    const action = url.match(/\/api\/bots\/(\d+)\/(start|pause|stop)$/)
    if (method === 'POST' && action) {
      const botId = Number(action[1])
      const target = bots.find((b) => b.id === botId)
      if (target) target.status = ACTION_STATUS[action[2]]
      return jsonResponse(target)
    }

    if (url.includes('/api/bots')) {
      return jsonResponse(bots)
    }

    return jsonResponse({}, { ok: false, status: 404 })
  })

  globalThis.fetch = fetchMock as typeof fetch
  return { fetchMock, calls }
}

function renderBotList() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <ToastProvider>
          <BotList />
        </ToastProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('BotList Start/Pause/Stop controls', () => {
  beforeEach(() => {
    // confirm() guards the Stop action; default to "yes" for tests.
    vi.stubGlobal('confirm', vi.fn(() => true))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('Start on a created bot POSTs to /start and the row becomes running', async () => {
    const { calls } = installFetchMock([makeBot({ id: 7, status: 'created' })])
    renderBotList()

    const startBtn = await screen.findByRole('button', { name: 'Start TestBot1' })
    fireEvent.click(startBtn)

    await waitFor(() =>
      expect(calls).toContainEqual({ url: '/api/bots/7/start', method: 'POST' }),
    )

    // The list refetches on success and the status badge updates — no reload.
    await waitFor(() => expect(screen.getByText('running')).toBeInTheDocument())
  })

  it('Stop on a running bot POSTs to /stop and the row becomes stopped', async () => {
    const { calls } = installFetchMock([makeBot({ id: 9, status: 'running' })])
    renderBotList()

    const stopBtn = await screen.findByRole('button', { name: 'Stop TestBot1' })
    fireEvent.click(stopBtn)

    await waitFor(() =>
      expect(calls).toContainEqual({ url: '/api/bots/9/stop', method: 'POST' }),
    )
    await waitFor(() => expect(screen.getByText('stopped')).toBeInTheDocument())
  })

  it('Pause on a running bot POSTs to /pause and the row becomes paused', async () => {
    const { calls } = installFetchMock([makeBot({ id: 4, status: 'running' })])
    renderBotList()

    const pauseBtn = await screen.findByRole('button', { name: 'Pause TestBot1' })
    fireEvent.click(pauseBtn)

    await waitFor(() =>
      expect(calls).toContainEqual({ url: '/api/bots/4/pause', method: 'POST' }),
    )
    await waitFor(() => expect(screen.getByText('paused')).toBeInTheDocument())
  })

  it('surfaces an error toast and keeps the row when the action API fails', async () => {
    // GET returns the bot; POST /start fails -> status must NOT change.
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = (init?.method ?? 'GET').toUpperCase()
      if (method === 'POST' && url.endsWith('/start')) {
        return jsonResponse({ detail: 'Engine offline' }, { ok: false, status: 500 })
      }
      return jsonResponse([makeBot({ id: 3, status: 'created' })])
    })
    globalThis.fetch = fetchMock as typeof fetch

    renderBotList()

    const startBtn = await screen.findByRole('button', { name: 'Start TestBot1' })
    fireEvent.click(startBtn)

    // Error toast shown (from the shared hook's onError).
    await waitFor(() => expect(screen.getByText('Engine offline')).toBeInTheDocument())
    // Status stays "created" — the failed action did not flip the row.
    const row = screen.getByText('TestBot1').closest('tr') as HTMLElement
    expect(within(row).getByText('created')).toBeInTheDocument()
  })
})
