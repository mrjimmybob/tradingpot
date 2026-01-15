/**
 * Connection status indicator for WebSocket.
 *
 * Shows:
 * - Green dot when connected
 * - Yellow dot when connecting
 * - Red dot when disconnected
 */
import { Wifi, WifiOff, Loader2 } from 'lucide-react'
import { useWebSocketContext } from '../contexts/WebSocketContext'

export function ConnectionStatus() {
  const { isConnected, isConnecting, reconnect } = useWebSocketContext()

  if (isConnecting) {
    return (
      <div className="flex items-center gap-2 text-paused">
        <Loader2 size={16} className="animate-spin" />
        <span className="text-sm hidden sm:inline">Connecting...</span>
      </div>
    )
  }

  if (!isConnected) {
    return (
      <button
        onClick={reconnect}
        className="flex items-center gap-2 text-loss hover:text-white transition-colors"
        title="Click to reconnect"
      >
        <WifiOff size={16} />
        <span className="text-sm hidden sm:inline">Disconnected</span>
      </button>
    )
  }

  return (
    <div className="flex items-center gap-2 text-profit">
      <Wifi size={16} />
      <span className="text-sm hidden sm:inline">Live</span>
    </div>
  )
}

export default ConnectionStatus
