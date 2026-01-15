import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Bot, ArrowLeft, ArrowRight, Check, AlertCircle, Loader2 } from 'lucide-react'
import { useToast } from '../components/Toast'
import ProgressIndicator from '../components/ProgressIndicator'

interface FormErrors {
  name?: string
  trading_pair?: string
  strategy?: string
  budget?: string
  stop_loss_percent?: string
  drawdown_limit_percent?: string
  daily_loss_limit?: string
  weekly_loss_limit?: string
  running_time_hours?: string
  [key: string]: string | undefined
}

interface Strategy {
  name: string
  display_name: string
  description: string
  parameters: Record<string, {
    type: string
    default: number | string | boolean
    min?: number
    max?: number
    description: string
  }>
}

interface TradingPair {
  symbol: string
  base: string
  quote: string
}

interface BotFormData {
  name: string
  trading_pair: string
  strategy: string
  strategy_params: Record<string, number | string | boolean>
  budget: number
  running_time_hours: number | null
  stop_loss_percent: number
  drawdown_limit_percent: number
  daily_loss_limit: number | null
  weekly_loss_limit: number | null
  is_dry_run: boolean
  compound_enabled: boolean
}

async function fetchStrategies(): Promise<Strategy[]> {
  const res = await fetch('/api/config/strategies')
  if (!res.ok) throw new Error('Failed to fetch strategies')
  return res.json()
}

async function fetchPairs(): Promise<TradingPair[]> {
  const res = await fetch('/api/config/pairs')
  if (!res.ok) throw new Error('Failed to fetch trading pairs')
  return res.json()
}

async function createBot(data: BotFormData) {
  const res = await fetch('/api/bots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}))
    if (errData.detail?.errors) {
      // Strategy validation errors
      throw new Error(errData.detail.errors.join('. '))
    }
    throw new Error(errData.detail || 'Failed to create bot')
  }
  return res.json()
}

const STEPS = ['Basic Info', 'Strategy', 'Risk Management', 'Review']

export default function CreateBot() {
  const navigate = useNavigate()
  const toast = useToast()
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState<BotFormData>({
    name: '',
    trading_pair: '',
    strategy: '',
    strategy_params: {},
    budget: 100,
    running_time_hours: null,
    stop_loss_percent: 5,
    drawdown_limit_percent: 10,
    daily_loss_limit: null,
    weekly_loss_limit: null,
    is_dry_run: true,
    compound_enabled: false,
  })
  const [errors, setErrors] = useState<FormErrors>({})
  const [touched, setTouched] = useState<Record<string, boolean>>({})

  // Clear error when field is corrected (#117)
  const clearError = useCallback((field: string) => {
    setErrors((prev) => {
      if (prev[field]) {
        const { [field]: _, ...rest } = prev
        return rest
      }
      return prev
    })
  }, [])

  const { data: strategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: fetchStrategies,
  })

  const { data: pairs = [] } = useQuery({
    queryKey: ['trading-pairs'],
    queryFn: fetchPairs,
  })

  const createMutation = useMutation({
    mutationFn: createBot,
    onSuccess: (data) => {
      toast.success(
        'Bot Created Successfully',
        `"${data.name}" is ready to trade ${data.trading_pair}${data.is_dry_run ? ' (Dry Run mode)' : ''}`
      )
      navigate(`/bots/${data.id}`)
    },
    onError: (err: Error) => {
      toast.error('Bot Creation Failed', err.message || 'Could not create bot. Please check your settings and try again.')
    },
  })

  const updateFormData = (updates: Partial<BotFormData>) => {
    setFormData((prev) => ({ ...prev, ...updates }))
    // Clear errors for updated fields (#117)
    Object.keys(updates).forEach((field) => {
      clearError(field)
    })
  }

  const markTouched = (field: string) => {
    setTouched((prev) => ({ ...prev, [field]: true }))
  }

  // #169: Validate to match server-side Pydantic validation
  const validateStep = (step: number): boolean => {
    const newErrors: FormErrors = {}

    if (step === 0) {
      // Server: name min_length=1, max_length=255
      if (!formData.name.trim()) {
        newErrors.name = 'Bot name is required'
      } else if (formData.name.length > 255) {
        newErrors.name = 'Bot name must be less than 255 characters'
      }
      // Server: trading_pair min_length=1, max_length=50
      if (!formData.trading_pair) {
        newErrors.trading_pair = 'Please select a trading pair'
      } else if (formData.trading_pair.length > 50) {
        newErrors.trading_pair = 'Trading pair must be less than 50 characters'
      }
    }

    if (step === 1) {
      // Server: strategy min_length=1, max_length=50
      if (!formData.strategy) {
        newErrors.strategy = 'Please select a strategy'
      } else if (formData.strategy.length > 50) {
        newErrors.strategy = 'Strategy name must be less than 50 characters'
      }

      // Validate strategy parameters against min/max constraints
      if (selectedStrategy) {
        Object.entries(selectedStrategy.parameters).forEach(([key, param]) => {
          const value = formData.strategy_params[key]
          if (value !== undefined && value !== null && param.type === 'number') {
            const numValue = Number(value)
            if (param.min !== undefined && numValue < param.min) {
              newErrors[`param_${key}`] = `${key.replace(/_/g, ' ')} must be at least ${param.min}`
            }
            if (param.max !== undefined && numValue > param.max) {
              newErrors[`param_${key}`] = `${key.replace(/_/g, ' ')} must be at most ${param.max}`
            }
          }
        })
      }
    }

    if (step === 2) {
      // Server: budget gt=0
      if (!formData.budget || formData.budget <= 0) {
        newErrors.budget = 'Budget must be greater than 0'
      }
      // Server: stop_loss_percent ge=0, le=100
      if (formData.stop_loss_percent < 0 || formData.stop_loss_percent > 100) {
        newErrors.stop_loss_percent = 'Stop loss must be between 0 and 100'
      }
      // Server: drawdown_limit_percent ge=0, le=100
      if (formData.drawdown_limit_percent < 0 || formData.drawdown_limit_percent > 100) {
        newErrors.drawdown_limit_percent = 'Drawdown limit must be between 0 and 100'
      }
      // Server: daily_loss_limit ge=0 (if provided)
      if (formData.daily_loss_limit !== null && formData.daily_loss_limit < 0) {
        newErrors.daily_loss_limit = 'Daily loss limit must be 0 or greater'
      }
      // Server: weekly_loss_limit ge=0 (if provided)
      if (formData.weekly_loss_limit !== null && formData.weekly_loss_limit < 0) {
        newErrors.weekly_loss_limit = 'Weekly loss limit must be 0 or greater'
      }
      // Server: running_time_hours should be positive if provided
      if (formData.running_time_hours !== null && formData.running_time_hours <= 0) {
        newErrors.running_time_hours = 'Running time must be greater than 0'
      }
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const selectedStrategy = strategies.find((s) => s.name === formData.strategy)

  // #169: canProceed matches validateStep and server-side validation
  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return formData.name.trim() && formData.trading_pair
      case 1:
        return formData.strategy
      case 2:
        // Server allows stop_loss_percent >= 0 and drawdown_limit_percent >= 0
        return formData.budget > 0 &&
               formData.stop_loss_percent >= 0 &&
               formData.drawdown_limit_percent >= 0
      default:
        return true
    }
  }

  const handleNext = () => {
    if (!validateStep(currentStep)) {
      return
    }
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      createMutation.mutate(formData)
    }
  }

  // Helper to get input class with error state
  const getInputClass = (field: string) => {
    const baseClass = 'w-full px-4 py-2 bg-gray-700 border rounded-lg text-white focus:outline-none focus:ring-2'
    if (errors[field] && touched[field]) {
      return `${baseClass} border-loss focus:ring-loss`
    }
    return `${baseClass} border-gray-600 focus:ring-accent`
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    } else {
      navigate('/bots')
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center gap-4 mb-8">
        <Bot size={32} className="text-accent" />
        <h2 className="text-2xl font-bold">Create New Bot</h2>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {STEPS.map((step, index) => (
          <div key={step} className="flex items-center">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                index < currentStep
                  ? 'bg-profit text-white'
                  : index === currentStep
                  ? 'bg-accent text-white'
                  : 'bg-gray-700 text-gray-400'
              }`}
            >
              {index < currentStep ? <Check size={16} /> : index + 1}
            </div>
            <span
              className={`ml-2 text-sm ${
                index <= currentStep ? 'text-white' : 'text-gray-400'
              }`}
            >
              {step}
            </span>
            {index < STEPS.length - 1 && (
              <div
                className={`w-12 h-0.5 mx-4 ${
                  index < currentStep ? 'bg-profit' : 'bg-gray-700'
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        {currentStep === 0 && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold mb-4">Basic Information</h3>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Bot Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => updateFormData({ name: e.target.value })}
                onBlur={() => markTouched('name')}
                className={getInputClass('name')}
                placeholder="My Trading Bot"
              />
              {errors.name && touched.name && (
                <p className="mt-1 text-sm text-loss flex items-center gap-1">
                  <AlertCircle size={14} />
                  {errors.name}
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Trading Pair
              </label>
              <select
                value={formData.trading_pair}
                onChange={(e) => updateFormData({ trading_pair: e.target.value })}
                onBlur={() => markTouched('trading_pair')}
                className={getInputClass('trading_pair')}
              >
                <option value="">Select a trading pair</option>
                {pairs.map((pair) => (
                  <option key={pair.symbol} value={pair.symbol}>
                    {pair.symbol}
                  </option>
                ))}
              </select>
              {errors.trading_pair && touched.trading_pair && (
                <p className="mt-1 text-sm text-loss flex items-center gap-1">
                  <AlertCircle size={14} />
                  {errors.trading_pair}
                </p>
              )}
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="dry_run"
                checked={formData.is_dry_run}
                onChange={(e) => updateFormData({ is_dry_run: e.target.checked })}
                className="w-4 h-4 text-accent bg-gray-700 border-gray-600 rounded focus:ring-accent"
              />
              <label htmlFor="dry_run" className="text-sm text-gray-300">
                Dry Run Mode (simulated trades, no real orders)
              </label>
            </div>
          </div>
        )}

        {currentStep === 1 && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold mb-4">Select Strategy</h3>

            {errors.strategy && (
              <p className="text-sm text-loss flex items-center gap-1 -mt-2 mb-4">
                <AlertCircle size={14} />
                {errors.strategy}
              </p>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {strategies.map((strategy) => (
                <button
                  key={strategy.name}
                  onClick={() =>
                    updateFormData({
                      strategy: strategy.name,
                      strategy_params: Object.fromEntries(
                        Object.entries(strategy.parameters).map(([key, param]) => [
                          key,
                          param.default,
                        ])
                      ),
                    })
                  }
                  className={`text-left p-4 rounded-lg border-2 transition-colors ${
                    formData.strategy === strategy.name
                      ? 'border-accent bg-accent/10'
                      : errors.strategy
                      ? 'border-loss/50 bg-gray-700 hover:border-loss'
                      : 'border-gray-600 bg-gray-700 hover:border-gray-500'
                  }`}
                >
                  <h4 className="font-medium text-white">{strategy.display_name}</h4>
                  <p className="text-sm text-gray-400 mt-1">{strategy.description}</p>
                </button>
              ))}
            </div>

            {selectedStrategy && Object.keys(selectedStrategy.parameters).length > 0 && (
              <div className="mt-6">
                <h4 className="text-md font-medium mb-4">Strategy Parameters</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(selectedStrategy.parameters).map(([key, param]) => {
                    const errorKey = `param_${key}`
                    const hasError = errors[errorKey] && touched[errorKey]
                    return (
                      <div key={key}>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          {key.replace(/_/g, ' ')}
                          {param.min !== undefined && param.max !== undefined && (
                            <span className="text-gray-500 font-normal"> ({param.min}-{param.max})</span>
                          )}
                        </label>
                        <input
                          type={param.type === 'number' ? 'number' : 'text'}
                          value={formData.strategy_params[key] ?? param.default}
                          onChange={(e) =>
                            updateFormData({
                              strategy_params: {
                                ...formData.strategy_params,
                                [key]:
                                  param.type === 'number'
                                    ? parseFloat(e.target.value)
                                    : e.target.value,
                              },
                            })
                          }
                          onBlur={() => markTouched(errorKey)}
                          min={param.min}
                          max={param.max}
                          className={`w-full px-4 py-2 bg-gray-700 border rounded-lg text-white focus:outline-none focus:ring-2 ${
                            hasError
                              ? 'border-loss focus:ring-loss'
                              : 'border-gray-600 focus:ring-accent'
                          }`}
                        />
                        <p className="text-xs text-gray-500 mt-1">{param.description}</p>
                        {hasError && (
                          <p className="mt-1 text-sm text-loss flex items-center gap-1">
                            <AlertCircle size={14} />
                            {errors[errorKey]}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {currentStep === 2 && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold mb-4">Risk Management</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Budget (USD)
                </label>
                <input
                  type="number"
                  value={formData.budget}
                  onChange={(e) =>
                    updateFormData({ budget: parseFloat(e.target.value) || 0 })
                  }
                  onBlur={() => markTouched('budget')}
                  min="0"
                  step="10"
                  className={getInputClass('budget')}
                />
                {errors.budget && touched.budget && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.budget}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Running Time (hours, empty = forever)
                </label>
                <input
                  type="number"
                  value={formData.running_time_hours ?? ''}
                  onChange={(e) =>
                    updateFormData({
                      running_time_hours: e.target.value
                        ? parseFloat(e.target.value)
                        : null,
                    })
                  }
                  onBlur={() => markTouched('running_time_hours')}
                  min="0"
                  className={getInputClass('running_time_hours')}
                  placeholder="Forever"
                />
                {errors.running_time_hours && touched.running_time_hours && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.running_time_hours}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Stop Loss (%)
                </label>
                <input
                  type="number"
                  value={formData.stop_loss_percent}
                  onChange={(e) =>
                    updateFormData({
                      stop_loss_percent: parseFloat(e.target.value) || 0,
                    })
                  }
                  onBlur={() => markTouched('stop_loss_percent')}
                  min="0"
                  max="100"
                  step="0.5"
                  className={getInputClass('stop_loss_percent')}
                />
                {errors.stop_loss_percent && touched.stop_loss_percent && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.stop_loss_percent}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Drawdown Limit (%)
                </label>
                <input
                  type="number"
                  value={formData.drawdown_limit_percent}
                  onChange={(e) =>
                    updateFormData({
                      drawdown_limit_percent: parseFloat(e.target.value) || 0,
                    })
                  }
                  onBlur={() => markTouched('drawdown_limit_percent')}
                  min="0"
                  max="100"
                  step="1"
                  className={getInputClass('drawdown_limit_percent')}
                />
                {errors.drawdown_limit_percent && touched.drawdown_limit_percent && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.drawdown_limit_percent}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Daily Loss Limit (USD, empty = none)
                </label>
                <input
                  type="number"
                  value={formData.daily_loss_limit ?? ''}
                  onChange={(e) =>
                    updateFormData({
                      daily_loss_limit: e.target.value
                        ? parseFloat(e.target.value)
                        : null,
                    })
                  }
                  onBlur={() => markTouched('daily_loss_limit')}
                  min="0"
                  className={getInputClass('daily_loss_limit')}
                  placeholder="No limit"
                />
                {errors.daily_loss_limit && touched.daily_loss_limit && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.daily_loss_limit}
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Weekly Loss Limit (USD, empty = none)
                </label>
                <input
                  type="number"
                  value={formData.weekly_loss_limit ?? ''}
                  onChange={(e) =>
                    updateFormData({
                      weekly_loss_limit: e.target.value
                        ? parseFloat(e.target.value)
                        : null,
                    })
                  }
                  onBlur={() => markTouched('weekly_loss_limit')}
                  min="0"
                  className={getInputClass('weekly_loss_limit')}
                  placeholder="No limit"
                />
                {errors.weekly_loss_limit && touched.weekly_loss_limit && (
                  <p className="mt-1 text-sm text-loss flex items-center gap-1">
                    <AlertCircle size={14} />
                    {errors.weekly_loss_limit}
                  </p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="compound"
                checked={formData.compound_enabled}
                onChange={(e) =>
                  updateFormData({ compound_enabled: e.target.checked })
                }
                className="w-4 h-4 text-accent bg-gray-700 border-gray-600 rounded focus:ring-accent"
              />
              <label htmlFor="compound" className="text-sm text-gray-300">
                Enable Compounding (add profits to budget)
              </label>
            </div>
          </div>
        )}

        {currentStep === 3 && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold mb-4">Review Configuration</h3>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-gray-400 mb-2">Basic Info</h4>
                <p>
                  <span className="text-gray-400">Name:</span>{' '}
                  <span className="text-white">{formData.name}</span>
                </p>
                <p>
                  <span className="text-gray-400">Pair:</span>{' '}
                  <span className="text-white">{formData.trading_pair}</span>
                </p>
                <p>
                  <span className="text-gray-400">Mode:</span>{' '}
                  <span className={formData.is_dry_run ? 'text-paused' : 'text-running'}>
                    {formData.is_dry_run ? 'Dry Run' : 'Live'}
                  </span>
                </p>
              </div>

              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-gray-400 mb-2">Strategy</h4>
                <p>
                  <span className="text-gray-400">Strategy:</span>{' '}
                  <span className="text-white">
                    {selectedStrategy?.display_name || formData.strategy}
                  </span>
                </p>
              </div>

              <div className="bg-gray-700 p-4 rounded-lg col-span-2">
                <h4 className="text-gray-400 mb-2">Risk Settings</h4>
                <div className="grid grid-cols-2 gap-2">
                  <p>
                    <span className="text-gray-400">Budget:</span>{' '}
                    <span className="text-white">${formData.budget}</span>
                  </p>
                  <p>
                    <span className="text-gray-400">Stop Loss:</span>{' '}
                    <span className="text-white">{formData.stop_loss_percent}%</span>
                  </p>
                  <p>
                    <span className="text-gray-400">Drawdown Limit:</span>{' '}
                    <span className="text-white">{formData.drawdown_limit_percent}%</span>
                  </p>
                  <p>
                    <span className="text-gray-400">Compounding:</span>{' '}
                    <span className="text-white">
                      {formData.compound_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </p>
                </div>
              </div>
            </div>

            {createMutation.isError && (
              <div className="bg-loss/20 text-loss px-4 py-3 rounded-lg">
                Failed to create bot. Please try again.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Navigation Buttons */}
      <div className="flex justify-between">
        <button
          onClick={handleBack}
          className="flex items-center gap-2 px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
        >
          <ArrowLeft size={18} />
          {currentStep === 0 ? 'Cancel' : 'Back'}
        </button>

        <button
          onClick={handleNext}
          disabled={!canProceed() || createMutation.isPending}
          className="flex items-center gap-2 px-6 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {createMutation.isPending ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Creating Bot...
            </>
          ) : currentStep === STEPS.length - 1 ? (
            <>
              Create Bot
              <Check size={18} />
            </>
          ) : (
            <>
              Next
              <ArrowRight size={18} />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
