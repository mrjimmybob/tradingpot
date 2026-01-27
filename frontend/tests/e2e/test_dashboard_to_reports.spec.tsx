/**
 * End-to-End Smoke Test: Dashboard → Reports Flow
 * 
 * Tests the main user journey from Dashboard to Reports using React Testing Library
 * and MemoryRouter. This is an integration test, not a full browser E2E test.
 * 
 * Flow: App boot → Dashboard → Navigate to Reports → Interact → Navigate back
 */

import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import App from '../../src/App';
import { WebSocketProvider } from '../../src/contexts/WebSocketContext';

// Mock WebSocket
class MockWebSocket {
  public readyState: number = WebSocket.OPEN;
  public onopen: ((ev: Event) => void) | null = null;
  public onclose: ((ev: CloseEvent) => void) | null = null;
  public onerror: ((ev: Event) => void) | null = null;
  public onmessage: ((ev: MessageEvent) => void) | null = null;

  constructor(public url: string) {
    setTimeout(() => {
      this.onopen?.(new Event('open'));
    }, 0);
  }

  send() {}
  close() {}
}

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
        cacheTime: 0,
      },
    },
  });

const renderApp = (initialRoute = '/') => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <WebSocketProvider>
        <MemoryRouter initialEntries={[initialRoute]}>
          <App />
        </MemoryRouter>
      </WebSocketProvider>
    </QueryClientProvider>
  );
};

// Mock data
const mockDashboardStats = {
  total_bots: 5,
  running_bots: 3,
  paused_bots: 1,
  stopped_bots: 1,
  total_pnl: 1234.56,
  active_trades: 2,
  dry_run_bots: 2,
};

const mockPnLHistory = [
  { timestamp: '2025-01-01T10:00:00Z', pnl: 100 },
  { timestamp: '2025-01-02T10:00:00Z', pnl: 150 },
  { timestamp: '2025-01-03T10:00:00Z', pnl: 200 },
];

const mockBots = [
  {
    id: 1,
    name: 'Bot Alpha',
    status: 'running',
    trading_pair: 'BTC/USDT',
    total_pnl: 500.0,
  },
  {
    id: 2,
    name: 'Bot Beta',
    status: 'paused',
    trading_pair: 'ETH/USDT',
    total_pnl: 734.56,
  },
];

const mockPnLReport = {
  entries: [
    {
      bot_id: 1,
      bot_name: 'Bot Alpha',
      trading_pair: 'BTC/USDT',
      strategy: 'momentum',
      total_pnl: 1500.0,
      win_count: 10,
      loss_count: 5,
      win_rate: 66.7,
      total_fees: 25.5,
    },
    {
      bot_id: 2,
      bot_name: 'Bot Beta',
      trading_pair: 'ETH/USDT',
      strategy: 'mean_reversion',
      total_pnl: -300.0,
      win_count: 5,
      loss_count: 10,
      win_rate: 33.3,
      total_fees: 15.0,
    },
  ],
  total_pnl: 1200.0,
  overall_win_rate: 50.0,
};

const mockTaxReport = {
  entries: [
    {
      date: '2025-01-15T10:00:00Z',
      trading_pair: 'BTC/USDT',
      purchase_price: 45000.0,
      sale_price: 48000.0,
      gain: 3000.0,
      token: 'BTC',
    },
  ],
  total_gains: 3000.0,
  year: 2025,
};

const mockFeeReport = {
  entries: [
    {
      bot_id: 1,
      bot_name: 'Bot Alpha',
      total_fees: 25.5,
      order_count: 50,
    },
  ],
  total_fees: 25.5,
};

const mockTaxSummary = {
  year: 2025,
  short_term_gains: 5000.0,
  long_term_gains: 2000.0,
  total_gains: 7000.0,
  wash_sales: 0,
  lot_count: 15,
  trade_count: 30,
};

const mockAuditLog = [
  {
    id: 1,
    timestamp: '2025-01-26T10:00:00Z',
    severity: 'info',
    source: 'trading_engine',
    bot_id: 1,
    message: 'Trade executed successfully',
    details: null,
  },
  {
    id: 2,
    timestamp: '2025-01-26T11:00:00Z',
    severity: 'warning',
    source: 'risk_management',
    bot_id: 1,
    message: 'Stop loss triggered',
    details: { loss_pct: 5.0 },
  },
];

const mockRealizedGains = [
  {
    id: 1,
    date: '2025-01-15',
    asset: 'BTC',
    amount: 0.5,
    realized_pnl: 2500.0,
    bot_id: 1,
  },
];

const mockEquityCurve = [
  { timestamp: '2025-01-01T00:00:00Z', equity: 10000.0 },
  { timestamp: '2025-01-15T00:00:00Z', equity: 11234.56 },
  { timestamp: '2025-01-26T00:00:00Z', equity: 12500.0 },
];

describe('E2E Smoke Test: Dashboard → Reports Flow', () => {
  let originalWebSocket: typeof WebSocket;

  beforeAll(() => {
    originalWebSocket = global.WebSocket;
    (global as any).WebSocket = MockWebSocket;
    WebSocket.CONNECTING = 0;
    WebSocket.OPEN = 1;
    WebSocket.CLOSING = 2;
    WebSocket.CLOSED = 3;
  });

  afterAll(() => {
    global.WebSocket = originalWebSocket;
  });

  beforeEach(() => {
    global.fetch = jest.fn();
    global.open = jest.fn();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.restoreAllMocks();
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe('1️⃣ App boot', () => {
    it('renders Dashboard as default route', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });
    });

    it('loads bot list successfully', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
      });
    });

    it('loads portfolio summary', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Running Bots')).toBeInTheDocument();
        expect(screen.getByText(/\$1234\.56/)).toBeInTheDocument();
      });
    });

    it('shows no crash on startup', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      const { container } = renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      expect(container).toBeInTheDocument();
      expect(container.querySelector('.error')).not.toBeInTheDocument();
    });
  });

  describe('2️⃣ Navigate to Reports', () => {
    it('click "Reports" navigation link', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });
    });

    it('route changes to /reports', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });
    });

    it('Reports page renders', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText('Reports')).toBeInTheDocument();
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
        expect(screen.getByText('Tax Export')).toBeInTheDocument();
        expect(screen.getByText('Fees')).toBeInTheDocument();
      });
    });
  });

  describe('3️⃣ Data rendering', () => {
    it('renders P&L report with all data', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });
    });

    it('renders tax report when tab clicked', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        if (url === '/api/reports/tax') {
          return Promise.resolve({
            ok: true,
            json: async () => mockTaxReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Total Capital Gains (2025)')).toBeInTheDocument();
      });
    });

    it('renders tables with rows', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        const table = screen.getByRole('table');
        expect(table).toBeInTheDocument();
        
        const rows = within(table).getAllByRole('row');
        expect(rows.length).toBeGreaterThan(1); // Header + data rows
      });
    });

    it('handles empty states safely', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              entries: [],
              total_pnl: 0,
              overall_win_rate: 0,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText(/No bots with P&L data yet/i)).toBeInTheDocument();
      });
    });
  });

  describe('4️⃣ Failure handling', () => {
    it('simulates API failure for one report', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.reject(new Error('Failed to fetch P&L report'));
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText(/No P&L data available/i)).toBeInTheDocument();
      });
    });

    it('other reports still render after one fails', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.reject(new Error('Failed to fetch P&L report'));
        }
        if (url === '/api/reports/tax') {
          return Promise.resolve({
            ok: true,
            json: async () => mockTaxReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText(/No P&L data available/i)).toBeInTheDocument();
      });

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Total Capital Gains (2025)')).toBeInTheDocument();
      });
    });

    it('error message shown only for failing section', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.reject(new Error('Failed to fetch P&L report'));
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      await waitFor(() => {
        const errorMessages = screen.getAllByText(/No P&L data available/i);
        expect(errorMessages).toHaveLength(1);
      });
    });

    it('app does not crash on error', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      const { container } = renderApp('/reports');

      await waitFor(() => {
        expect(screen.getByText('Reports')).toBeInTheDocument();
      });

      expect(container).toBeInTheDocument();
    });
  });

  describe('5️⃣ User interactions', () => {
    it('export button calls export API', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        if (url === '/api/reports/tax') {
          return Promise.resolve({
            ok: true,
            json: async () => mockTaxReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Export CSV')).toBeInTheDocument();
      });

      const exportButton = screen.getByText('Export CSV');
      fireEvent.click(exportButton);

      expect(global.open).toHaveBeenCalledWith(
        '/api/reports/tax?format=csv',
        '_blank'
      );
    });

    it('switches tabs between reports', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        if (url === '/api/reports/tax') {
          return Promise.resolve({
            ok: true,
            json: async () => mockTaxReport,
          });
        }
        if (url === '/api/reports/fees') {
          return Promise.resolve({
            ok: true,
            json: async () => mockFeeReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/reports');

      // Start on P&L
      await waitFor(() => {
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });

      // Switch to Tax
      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Total Capital Gains (2025)')).toBeInTheDocument();
      });

      // Switch to Fees
      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText('Total Fees Paid')).toBeInTheDocument();
      });

      // Switch back to P&L
      const pnlTab = screen.getByText('P&L Report');
      fireEvent.click(pnlTab);

      await waitFor(() => {
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });
    });
  });

  describe('6️⃣ Navigation safety', () => {
    it('navigate back to Dashboard', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });

      const dashboardLink = screen.getByText('Dashboard');
      fireEvent.click(dashboardLink);

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Running Bots')).toBeInTheDocument();
      });
    });

    it('dashboard state remains valid after navigation', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
      });

      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });

      const dashboardLink = screen.getByText('Dashboard');
      fireEvent.click(dashboardLink);

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
        expect(screen.getByText(/\$1234\.56/)).toBeInTheDocument();
      });
    });

    it('navigate back to Reports again', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });

      const dashboardLink = screen.getByText('Dashboard');
      fireEvent.click(dashboardLink);

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Reports'));

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });
    });

    it('reports refetch data on navigation', async () => {
      let pnlCallCount = 0;

      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          pnlCallCount++;
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      renderApp('/');

      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(pnlCallCount).toBeGreaterThan(0);
      });

      const firstCallCount = pnlCallCount;

      const dashboardLink = screen.getByText('Dashboard');
      fireEvent.click(dashboardLink);

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Reports'));

      await waitFor(() => {
        expect(pnlCallCount).toBe(firstCallCount + 1);
      });
    });
  });

  describe('🧪 Integration: Full happy path', () => {
    it('completes full user journey without errors', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url === '/api/stats') {
          return Promise.resolve({
            ok: true,
            json: async () => mockDashboardStats,
          });
        }
        if (url === '/api/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLHistory,
          });
        }
        if (url === '/api/bots?limit=5') {
          return Promise.resolve({
            ok: true,
            json: async () => mockBots,
          });
        }
        if (url === '/api/reports/pnl') {
          return Promise.resolve({
            ok: true,
            json: async () => mockPnLReport,
          });
        }
        if (url === '/api/reports/tax') {
          return Promise.resolve({
            ok: true,
            json: async () => mockTaxReport,
          });
        }
        if (url === '/api/reports/fees') {
          return Promise.resolve({
            ok: true,
            json: async () => mockFeeReport,
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        });
      });

      // 1. Start on Dashboard
      renderApp('/');

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
      });

      // 2. Navigate to Reports
      const reportsLink = screen.getByText('Reports');
      fireEvent.click(reportsLink);

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });

      // 3. View Tax report
      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Total Capital Gains (2025)')).toBeInTheDocument();
      });

      // 4. Export tax data
      const exportButton = screen.getByText('Export CSV');
      fireEvent.click(exportButton);

      expect(global.open).toHaveBeenCalledWith(
        '/api/reports/tax?format=csv',
        '_blank'
      );

      // 5. View Fees report
      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText('Total Fees Paid')).toBeInTheDocument();
      });

      // 6. Navigate back to Dashboard
      const dashboardLink = screen.getByText('Dashboard');
      fireEvent.click(dashboardLink);

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
      });
    });
  });
});
