/**
 * Central API client.
 *
 * All backend calls go through apiFetch so the API token (when configured
 * on the backend via TRADINGBOT_API_TOKEN) is attached consistently.
 * The token is stored locally and can be set on the Settings page.
 */

const TOKEN_STORAGE_KEY = 'tradingbot_api_token'

export function getApiToken(): string {
  return localStorage.getItem(TOKEN_STORAGE_KEY) ?? ''
}

export function setApiToken(token: string): void {
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token)
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
  }
}

/**
 * Drop-in replacement for fetch() that attaches the Authorization header
 * and converts 401 responses into a descriptive error so pages surface
 * a clear message instead of a generic failure.
 */
export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<Response> {
  const token = getApiToken()
  const headers = new Headers(init.headers)
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(input, { ...init, headers })

  if (response.status === 401) {
    throw new Error(
      'API authentication failed: token missing or invalid. Set your API token on the Settings page.'
    )
  }

  return response
}

/** Build the WebSocket URL, appending the API token when one is stored. */
export function buildWsUrl(base: string): string {
  const token = getApiToken()
  if (!token) return base
  const separator = base.includes('?') ? '&' : '?'
  return `${base}${separator}token=${encodeURIComponent(token)}`
}
