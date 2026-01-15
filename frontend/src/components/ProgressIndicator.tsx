import { Loader2 } from 'lucide-react'

interface ProgressIndicatorProps {
  /** The message to display */
  message?: string
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Whether to show as overlay */
  overlay?: boolean
  /** Progress percentage (0-100) for determinate progress, undefined for indeterminate */
  progress?: number
}

export default function ProgressIndicator({
  message = 'Please wait...',
  size = 'md',
  overlay = false,
  progress,
}: ProgressIndicatorProps) {
  const sizeClasses = {
    sm: { spinner: 16, text: 'text-sm', bar: 'h-1' },
    md: { spinner: 24, text: 'text-base', bar: 'h-2' },
    lg: { spinner: 32, text: 'text-lg', bar: 'h-3' },
  }

  const { spinner, text, bar } = sizeClasses[size]

  const content = (
    <div className="flex flex-col items-center gap-3">
      <Loader2 size={spinner} className="animate-spin text-accent" />
      <span className={`${text} text-gray-300`}>{message}</span>
      {progress !== undefined && (
        <div className="w-48 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`${bar} bg-accent transition-all duration-300 ease-out`}
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  )

  if (overlay) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
        <div className="bg-gray-800 rounded-lg p-8 shadow-xl">
          {content}
        </div>
      </div>
    )
  }

  return content
}

/** Inline loading spinner for buttons and small areas */
export function LoadingSpinner({ size = 16, className = '' }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={`animate-spin ${className}`} />
}

/** Full-page loading state */
export function PageLoader({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="flex items-center justify-center h-64">
      <ProgressIndicator message={message} size="lg" />
    </div>
  )
}

/** Operation status indicator for async operations */
interface OperationStatusProps {
  isLoading: boolean
  loadingMessage?: string
  children: React.ReactNode
}

export function OperationStatus({ isLoading, loadingMessage, children }: OperationStatusProps) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-gray-400">
        <LoadingSpinner size={14} />
        <span className="text-sm">{loadingMessage || 'Processing...'}</span>
      </div>
    )
  }
  return <>{children}</>
}
