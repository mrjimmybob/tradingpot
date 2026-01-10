import { Outlet, Link, useLocation } from 'react-router-dom'
import { useState } from 'react'
import {
  LayoutDashboard,
  Bot,
  FileBarChart,
  Settings,
  Power,
  Menu,
  X,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Bots', href: '/bots', icon: Bot },
  { name: 'Reports', href: '/reports', icon: FileBarChart },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export default function Layout() {
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [globalPnL] = useState(0) // TODO: Fetch from API

  const handleGlobalKillSwitch = async () => {
    if (confirm('Are you sure you want to activate the global kill switch? This will stop all running bots.')) {
      // TODO: Call API
      console.log('Global kill switch activated')
    }
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center justify-between px-4 h-16">
          <div className="flex items-center gap-4">
            <button
              className="lg:hidden p-2 text-gray-400 hover:text-white focus-ring rounded"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
            <h1 className="text-xl font-bold text-white">TradingBot</h1>
          </div>

          <div className="flex items-center gap-6">
            {/* Global P&L */}
            <div className="hidden sm:block">
              <span className="text-gray-400 text-sm mr-2">Global P&L:</span>
              <span
                className={`font-mono-numbers font-bold ${
                  globalPnL >= 0 ? 'text-profit' : 'text-loss'
                }`}
              >
                {globalPnL >= 0 ? '+' : ''}${globalPnL.toFixed(2)}
              </span>
            </div>

            {/* Global Kill Switch */}
            <button
              onClick={handleGlobalKillSwitch}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors focus-ring"
              aria-label="Global kill switch"
            >
              <Power size={18} />
              <span className="hidden sm:inline">Kill All</span>
            </button>
          </div>
        </div>
      </header>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside
          className={`fixed lg:static inset-y-0 left-0 z-40 w-64 bg-gray-800 border-r border-gray-700 transform transition-transform duration-200 ease-in-out lg:transform-none ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          } pt-16 lg:pt-0`}
        >
          <nav className="p-4 space-y-2">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href ||
                (item.href !== '/' && location.pathname.startsWith(item.href))

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors focus-ring ${
                    isActive
                      ? 'bg-accent text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <item.icon size={20} />
                  {item.name}
                </Link>
              )
            })}
          </nav>
        </aside>

        {/* Backdrop for mobile sidebar */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/50 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main content */}
        <main className="flex-1 p-6 lg:ml-0 min-h-[calc(100vh-4rem)]">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
