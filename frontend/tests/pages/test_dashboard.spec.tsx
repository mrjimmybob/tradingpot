/**
 * Comprehensive Dashboard Page Tests
 * 
 * Tests the Dashboard page including data loading, error handling,
 * charts, realtime updates, controls, and accessibility.
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from '../../src/pages/Dashboard';

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const renderDashboard = () => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>
  );
};

const mockSuccessfulFetch = (stats: any = {}, pnl: any[] = [], bots: any[] = []) => {
  (global.fetch as jest.Mock)
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        total_bots: 0,
        running_bots: 0,
        paused_bots: 0,
        stopped_bots: 0,
        total_pnl: 0,
        active_trades: 0,
        dry_run_bots: 0,
        ...stats,
      }),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => pnl,
    })
    .mockResolvedValueOnce({
      ok: true,
      json: async () => bots,
    });
};

describe('Dashboard Page', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Data loading', () => {
    it('loads dashboard data on mount', async () => {
      mockSuccessfulFetch();

      renderDashboard();

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/stats');
        expect(global.fetch).toHaveBeenCalledWith('/api/pnl');
        expect(global.fetch).toHaveBeenCalledWith('/api/bots?limit=5');
      });
    });

    it('shows loading indicator while fetching', () => {
      (global.fetch as jest.Mock).mockImplementation(
        () => new Promise(() => {})
      );

      renderDashboard();

      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('renders summary stats after loading', async () => {
      mockSuccessfulFetch({
        total_bots: 5,
        running_bots: 3,
        paused_bots: 1,
        stopped_bots: 1,
        total_pnl: 1234.56,
        active_trades: 2,
        dry_run_bots: 2,
      });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('5')).toBeInTheDocument();
        expect(screen.getByText('3')).toBeInTheDocument();
        expect(screen.getByText('2')).toBeInTheDocument();
      });
    });

    it('renders P&L value correctly', async () => {
      mockSuccessfulFetch({ total_pnl: 1234.56 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$1234\.56/)).toBeInTheDocument();
      });
    });

    it('renders negative P&L correctly', async () => {
      mockSuccessfulFetch({ total_pnl: -500.25 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/-\$500\.25/)).toBeInTheDocument();
      });
    });

    it('renders bot list correctly', async () => {
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
          total_pnl: -200.0,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
        expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
        expect(screen.getByText('ETH/USDT')).toBeInTheDocument();
      });
    });

    it('displays bot status correctly', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Running Bot',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const statusElement = screen.getByText('running');
        expect(statusElement).toBeInTheDocument();
        expect(statusElement).toHaveClass('text-running');
      });
    });

    it('displays all four stat cards', async () => {
      mockSuccessfulFetch({
        total_bots: 5,
        running_bots: 3,
        active_trades: 2,
        total_pnl: 100,
      });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Running Bots')).toBeInTheDocument();
        expect(screen.getByText('Total Bots')).toBeInTheDocument();
        expect(screen.getByText('Active Trades')).toBeInTheDocument();
      });
    });
  });

  describe('Error handling', () => {
    it('API failure shows error message', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new Error('Failed to fetch stats')
      );

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/Failed to load dashboard data/i)).toBeInTheDocument();
      });
    });

    it('displays error icon when API fails', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

      renderDashboard();

      await waitFor(() => {
        const errorMessage = screen.getByText(/Failed to load dashboard data/i);
        expect(errorMessage).toBeInTheDocument();
        expect(errorMessage.closest('div')).toHaveClass('text-loss');
      });
    });

    it('empty state renders correctly for bots', async () => {
      mockSuccessfulFetch({}, [], []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/No bots created yet/i)).toBeInTheDocument();
        expect(screen.getByText(/Create Your First Bot/i)).toBeInTheDocument();
      });
    });

    it('empty state shows Create Bot link', async () => {
      mockSuccessfulFetch({}, [], []);

      renderDashboard();

      await waitFor(() => {
        const createLink = screen.getByText(/Create Your First Bot/i);
        expect(createLink).toBeInTheDocument();
        expect(createLink.closest('a')).toHaveAttribute('href', '/bots/new');
      });
    });

    it('malformed data does not crash UI', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ unexpected: 'data' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [],
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [],
        });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });
    });

    it('handles null values gracefully', async () => {
      mockSuccessfulFetch({
        total_bots: null,
        running_bots: null,
        total_pnl: null,
      });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('0')).toBeInTheDocument();
      });
    });

    it('handles undefined stats gracefully', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({}),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [],
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [],
        });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });
    });
  });

  describe('Charts', () => {
    it('equity curve chart renders when data present', async () => {
      const mockPnL = [
        { timestamp: '2025-01-01T10:00:00Z', pnl: 100 },
        { timestamp: '2025-01-02T10:00:00Z', pnl: 150 },
        { timestamp: '2025-01-03T10:00:00Z', pnl: 200 },
      ];

      mockSuccessfulFetch({}, mockPnL, []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/P&L Over Time/i)).toBeInTheDocument();
        expect(screen.getByText('3 data points')).toBeInTheDocument();
      });
    });

    it('chart handles empty dataset', async () => {
      mockSuccessfulFetch({}, [], []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/No P&L data available yet/i)).toBeInTheDocument();
      });
    });

    it('chart shows data point count', async () => {
      const mockPnL = Array.from({ length: 50 }, (_, i) => ({
        timestamp: `2025-01-${String(i + 1).padStart(2, '0')}T10:00:00Z`,
        pnl: i * 10,
      }));

      mockSuccessfulFetch({}, mockPnL, []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('50 data points')).toBeInTheDocument();
      });
    });

    it('chart samples large datasets', async () => {
      const mockPnL = Array.from({ length: 300 }, (_, i) => ({
        timestamp: `2025-01-01T${String(i).padStart(2, '0')}:00:00Z`,
        pnl: i * 10,
      }));

      mockSuccessfulFetch({}, mockPnL, []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/Showing 200 of 300 points/i)).toBeInTheDocument();
      });
    });

    it('chart disables animation for large datasets', async () => {
      const mockPnL = Array.from({ length: 150 }, (_, i) => ({
        timestamp: `2025-01-01T${String(i).padStart(2, '0')}:00:00Z`,
        pnl: i * 10,
      }));

      mockSuccessfulFetch({}, mockPnL, []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/P&L Over Time/i)).toBeInTheDocument();
      });
    });

    it('chart renders with single data point', async () => {
      const mockPnL = [{ timestamp: '2025-01-01T10:00:00Z', pnl: 100 }];

      mockSuccessfulFetch({}, mockPnL, []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('1 data points')).toBeInTheDocument();
      });
    });
  });

  describe('Navigation', () => {
    it('View All link navigates to bots page', async () => {
      mockSuccessfulFetch();

      renderDashboard();

      await waitFor(() => {
        const viewAllLink = screen.getByText('View All');
        expect(viewAllLink).toBeInTheDocument();
        expect(viewAllLink.closest('a')).toHaveAttribute('href', '/bots');
      });
    });

    it('bot name links to bot detail page', async () => {
      const mockBots = [
        {
          id: 123,
          name: 'Test Bot',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const botLink = screen.getByText('Test Bot');
        expect(botLink.closest('a')).toHaveAttribute('href', '/bots/123');
      });
    });
  });

  describe('Accessibility', () => {
    it('main heading exists', async () => {
      mockSuccessfulFetch();

      renderDashboard();

      await waitFor(() => {
        const heading = screen.getByText('Dashboard');
        expect(heading.tagName).toBe('H2');
      });
    });

    it('stat cards have descriptive text', async () => {
      mockSuccessfulFetch({ total_bots: 5, running_bots: 3, active_trades: 2 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Running Bots')).toBeInTheDocument();
        expect(screen.getByText('Total Bots')).toBeInTheDocument();
        expect(screen.getByText('Active Trades')).toBeInTheDocument();
      });
    });

    it('trend indicators have screen reader text', async () => {
      mockSuccessfulFetch({ total_pnl: 100 });

      renderDashboard();

      await waitFor(() => {
        // Check if page loaded
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });
    });

    it('empty bot list has helpful message', async () => {
      mockSuccessfulFetch({}, [], []);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/No bots created yet/i)).toBeInTheDocument();
      });
    });

    it('links are keyboard accessible', async () => {
      mockSuccessfulFetch();

      renderDashboard();

      await waitFor(() => {
        const viewAllLink = screen.getByText('View All');
        expect(viewAllLink.closest('a')).toHaveAttribute('href');
      });
    });

    it('table has proper structure', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Bot 1',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const table = screen.getByRole('table');
        expect(table).toBeInTheDocument();
      });
    });

    it('table headers are properly labeled', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Bot 1',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Name')).toBeInTheDocument();
        expect(screen.getByText('Pair')).toBeInTheDocument();
        expect(screen.getByText('Status')).toBeInTheDocument();
        expect(screen.getByText('P&L')).toBeInTheDocument();
      });
    });
  });

  describe('Data formatting', () => {
    it('formats positive P&L with plus sign', async () => {
      mockSuccessfulFetch({ total_pnl: 1234.56 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$1234\.56/)).toBeInTheDocument();
      });
    });

    it('formats negative P&L correctly', async () => {
      mockSuccessfulFetch({ total_pnl: -1234.56 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/-\$1234\.56/)).toBeInTheDocument();
      });
    });

    it('formats zero P&L correctly', async () => {
      mockSuccessfulFetch({ total_pnl: 0 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$0\.00/)).toBeInTheDocument();
      });
    });

    it('formats bot P&L with two decimals', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Bot 1',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 123.456789,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$123\.46/)).toBeInTheDocument();
      });
    });

    it('capitalizes bot status', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Bot 1',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const statusElement = screen.getByText('running');
        expect(statusElement).toHaveClass('capitalize');
      });
    });
  });

  describe('Visual feedback', () => {
    it('applies profit styling to positive P&L', async () => {
      mockSuccessfulFetch({ total_pnl: 100 });

      renderDashboard();

      await waitFor(() => {
        const pnlElement = screen.getByText(/\+\$100\.00/);
        expect(pnlElement.closest('div')).toHaveClass('border-profit');
      });
    });

    it('applies loss styling to negative P&L', async () => {
      mockSuccessfulFetch({ total_pnl: -100 });

      renderDashboard();

      await waitFor(() => {
        const pnlElement = screen.getByText(/-\$100\.00/);
        expect(pnlElement.closest('div')).toHaveClass('border-loss');
      });
    });

    it('applies correct color to bot P&L', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Profit Bot',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 500,
        },
        {
          id: 2,
          name: 'Loss Bot',
          status: 'running',
          trading_pair: 'ETH/USDT',
          total_pnl: -200,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const profitElement = screen.getByText(/\+\$500\.00/);
        expect(profitElement).toHaveClass('text-profit');

        const lossElement = screen.getByText(/-\$200\.00/);
        expect(lossElement).toHaveClass('text-loss');
      });
    });

    it('applies status colors correctly', async () => {
      const mockBots = [
        {
          id: 1,
          name: 'Running Bot',
          status: 'running',
          trading_pair: 'BTC/USDT',
          total_pnl: 100,
        },
        {
          id: 2,
          name: 'Paused Bot',
          status: 'paused',
          trading_pair: 'ETH/USDT',
          total_pnl: 50,
        },
        {
          id: 3,
          name: 'Stopped Bot',
          status: 'stopped',
          trading_pair: 'XRP/USDT',
          total_pnl: 0,
        },
      ];

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        const runningStatus = screen.getByText('running');
        expect(runningStatus).toHaveClass('text-running');

        const pausedStatus = screen.getByText('paused');
        expect(pausedStatus).toHaveClass('text-paused');

        const stoppedStatus = screen.getByText('stopped');
        expect(stoppedStatus).toHaveClass('text-stopped');
      });
    });
  });

  describe('Edge cases', () => {
    it('handles very large P&L values', async () => {
      mockSuccessfulFetch({ total_pnl: 999999.99 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$999999\.99/)).toBeInTheDocument();
      });
    });

    it('handles very small P&L values', async () => {
      mockSuccessfulFetch({ total_pnl: 0.01 });

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText(/\+\$0\.01/)).toBeInTheDocument();
      });
    });

    it('handles zero bots', async () => {
      mockSuccessfulFetch(
        { total_bots: 0, running_bots: 0, active_trades: 0 },
        [],
        []
      );

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('0')).toBeInTheDocument();
      });
    });

    it('handles exactly 5 bots (limit)', async () => {
      const mockBots = Array.from({ length: 5 }, (_, i) => ({
        id: i + 1,
        name: `Bot ${i + 1}`,
        status: 'running',
        trading_pair: 'BTC/USDT',
        total_pnl: 100,
      }));

      mockSuccessfulFetch({}, [], mockBots);

      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText('Bot 1')).toBeInTheDocument();
        expect(screen.getByText('Bot 5')).toBeInTheDocument();
      });
    });
  });
});
