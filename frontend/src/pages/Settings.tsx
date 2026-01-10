import { useQuery } from '@tanstack/react-query'
import { Settings as SettingsIcon, FileCode, Server, Shield, Database } from 'lucide-react'

interface ConfigFile {
  name: string
  content: string
}

async function fetchConfig(): Promise<Record<string, ConfigFile>> {
  const res = await fetch('/api/config')
  if (!res.ok) throw new Error('Failed to fetch config')
  return res.json()
}

export default function Settings() {
  const { data: config, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: fetchConfig,
  })

  const configSections = [
    {
      id: 'exchanges',
      name: 'Exchange Configuration',
      icon: Server,
      description: 'API keys and exchange settings (sensitive data hidden)',
    },
    {
      id: 'email',
      name: 'Email Alerts',
      icon: Shield,
      description: 'SMTP settings for alert notifications',
    },
    {
      id: 'data_sources',
      name: 'Data Sources',
      icon: Database,
      description: 'External data source configuration',
    },
    {
      id: 'defaults',
      name: 'Default Parameters',
      icon: FileCode,
      description: 'Default strategy and risk parameters',
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <SettingsIcon size={32} className="text-accent" />
        <h2 className="text-2xl font-bold">Settings</h2>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-400">
          Configuration is managed through YAML files in the <code className="text-accent">config/</code> directory.
          This page provides a read-only view of the current configuration.
        </p>
        <p className="text-gray-400 mt-2">
          To modify settings, edit the YAML files directly and restart the server.
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
        </div>
      ) : (
        <div className="space-y-4">
          {configSections.map((section) => {
            const configData = config?.[section.id]
            return (
              <div key={section.id} className="bg-gray-800 rounded-lg overflow-hidden">
                <div className="flex items-center gap-4 p-4 border-b border-gray-700">
                  <section.icon size={24} className="text-accent" />
                  <div>
                    <h3 className="font-semibold">{section.name}</h3>
                    <p className="text-sm text-gray-400">{section.description}</p>
                  </div>
                </div>
                <div className="p-4">
                  {configData ? (
                    <pre className="text-sm text-gray-300 font-mono bg-gray-900 p-4 rounded-lg overflow-x-auto">
                      {configData.content}
                    </pre>
                  ) : (
                    <p className="text-gray-500 italic">Configuration file not found</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Server Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Backend URL:</span>{' '}
            <code className="text-accent">http://localhost:8000</code>
          </div>
          <div>
            <span className="text-gray-400">Frontend URL:</span>{' '}
            <code className="text-accent">http://localhost:5173</code>
          </div>
          <div>
            <span className="text-gray-400">Database:</span>{' '}
            <code className="text-accent">SQLite (tradingbot.db)</code>
          </div>
          <div>
            <span className="text-gray-400">Log Directory:</span>{' '}
            <code className="text-accent">logs/</code>
          </div>
        </div>
      </div>
    </div>
  )
}
