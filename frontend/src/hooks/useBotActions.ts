/**
 * Shared bot lifecycle controls (start / pause / stop).
 *
 * Both the bots list (/bots) and the bot detail page (/bots/:id) need the same
 * Start/Pause/Stop behaviour. Keeping a single implementation here means there
 * is exactly ONE start/stop code path: the same endpoints, the same error
 * handling, the same toast feedback, and the same cache invalidation. The bug
 * this fixes was that the list page rendered the control buttons with no
 * handler at all, so they hit none of this.
 *
 * The mutations take the bot id (and optional name, for the toast) at
 * `mutate()` time so a single hook instance can drive every row in the list as
 * well as the single bot on the detail page.
 */
import { apiFetch } from '../lib/api'
import { useMutation, useQueryClient, type UseMutationResult } from '@tanstack/react-query'
import { useToast } from '../components/Toast'

export interface BotActionVars {
  id: number | string
  /** Bot name, used only to personalise the success toast. */
  name?: string
}

type BotAction = 'start' | 'pause' | 'stop'

async function postBotAction(id: number | string, action: BotAction, failMsg: string) {
  const res = await apiFetch(`/api/bots/${id}/${action}`, { method: 'POST' })
  if (!res.ok) {
    // Tolerate a non-JSON/empty error body so a failure surfaces a clear
    // message instead of an opaque JSON.parse error.
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || failMsg)
  }
  return res.json()
}

export interface BotActions {
  start: UseMutationResult<unknown, Error, BotActionVars>
  pause: UseMutationResult<unknown, Error, BotActionVars>
  stop: UseMutationResult<unknown, Error, BotActionVars>
}

export function useBotActions(): BotActions {
  const queryClient = useQueryClient()
  const toast = useToast()

  const invalidate = (id: number | string) => {
    // Refresh both the single-bot detail view and the bots list so whichever
    // page triggered the action — and the other — reflect the new status
    // automatically, with no manual refetch and no page reload.
    queryClient.invalidateQueries({ queryKey: ['bot', String(id)] })
    queryClient.invalidateQueries({ queryKey: ['bots'] })
  }

  const labelFor = (name?: string) => (name ? `"${name}"` : 'The bot')

  const start = useMutation<unknown, Error, BotActionVars>({
    mutationFn: (vars) => postBotAction(vars.id, 'start', 'Failed to start bot'),
    onSuccess: (_data, vars) => {
      invalidate(vars.id)
      toast.success('Bot Started', `${labelFor(vars.name)} is now running and executing trades`)
    },
    onError: (err) => toast.error('Start Failed', err.message || 'Could not start the bot. Please try again.'),
  })

  const pause = useMutation<unknown, Error, BotActionVars>({
    mutationFn: (vars) => postBotAction(vars.id, 'pause', 'Failed to pause bot'),
    onSuccess: (_data, vars) => {
      invalidate(vars.id)
      toast.info('Bot Paused', `${labelFor(vars.name)} has been paused. No new trades will be executed.`)
    },
    onError: (err) => toast.error('Pause Failed', err.message || 'Could not pause the bot. Please try again.'),
  })

  const stop = useMutation<unknown, Error, BotActionVars>({
    mutationFn: (vars) => postBotAction(vars.id, 'stop', 'Failed to stop bot'),
    onSuccess: (_data, vars) => {
      invalidate(vars.id)
      toast.info('Bot Stopped', `${labelFor(vars.name)} has been stopped completely.`)
    },
    onError: (err) => toast.error('Stop Failed', err.message || 'Could not stop the bot. Please try again.'),
  })

  return { start, pause, stop }
}
