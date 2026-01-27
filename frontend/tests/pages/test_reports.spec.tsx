/**
 * Comprehensive Reports Page Tests
 * 
 * Tests the Reports page including P&L, Tax, and Fee reports with
 * data loading, tab switching, error handling, and accessibility.
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Reports from '../../src/pages/Reports';

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const renderReports = () => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <Reports />
    </QueryClientProvider>
  );
};

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
    {
      date: '2025-02-20T14:30:00Z',
      trading_pair: 'ETH/USDT',
      purchase_price: 3000.0,
      sale_price: 2800.0,
      gain: -200.0,
      token: 'ETH',
    },
  ],
  total_gains: 2800.0,
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
    {
      bot_id: 2,
      bot_name: 'Bot Beta',
      total_fees: 15.0,
      order_count: 30,
    },
  ],
  total_fees: 40.5,
};

describe('Reports Page', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
    global.open = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Data loading', () => {
    it('fetches P&L report on mount (default tab)', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/reports/pnl');
      });
    });

    it('does not fetch tax report until tab is active', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/reports/pnl');
      });

      expect(global.fetch).not.toHaveBeenCalledWith('/api/reports/tax');
    });

    it('fetches tax report when tax tab is clicked', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('P&L Report')).toBeInTheDocument();
      });

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/reports/tax');
      });
    });

    it('fetches fee report when fees tab is clicked', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockFeeReport,
        });

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/reports/fees');
      });
    });

    it('shows loading indicator while fetching P&L', () => {
      (global.fetch as jest.Mock).mockImplementation(
        () => new Promise(() => {})
      );

      renderReports();

      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('shows loading indicator while fetching tax report', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockImplementationOnce(() => new Promise(() => {}));

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
      });

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        const spinner = document.querySelector('.animate-spin');
        expect(spinner).toBeInTheDocument();
      });
    });

    it('renders report sections once loaded', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });
    });
  });

  describe('P&L Report section', () => {
    it('renders P&L summary cards correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Total P&L')).toBeInTheDocument();
        expect(screen.getByText('Overall Win Rate')).toBeInTheDocument();
        expect(screen.getByText('Total Trades')).toBeInTheDocument();
      });
    });

    it('displays total P&L with correct formatting', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/\+\$1200\.00/)).toBeInTheDocument();
      });
    });

    it('handles negative P&L correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          ...mockPnLReport,
          total_pnl: -500.0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/-\$500\.00/)).toBeInTheDocument();
      });
    });

    it('displays win rate percentage', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('50.0%')).toBeInTheDocument();
        expect(screen.getByText('15W / 15L')).toBeInTheDocument();
      });
    });

    it('calculates total trades correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('30')).toBeInTheDocument();
      });
    });

    it('renders bot P&L table with all entries', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
        expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
        expect(screen.getByText('ETH/USDT')).toBeInTheDocument();
      });
    });

    it('formats strategy names correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('momentum')).toBeInTheDocument();
        expect(screen.getByText('mean reversion')).toBeInTheDocument();
      });
    });

    it('displays fees in loss color', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        const feeElements = screen.getAllByText(/\$25\.50|\$15\.00/);
        feeElements.forEach(el => {
          expect(el).toHaveClass('text-loss');
        });
      });
    });

    it('handles empty P&L entries', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          entries: [],
          total_pnl: 0,
          overall_win_rate: 0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/No bots with P&L data yet/i)).toBeInTheDocument();
      });
    });

    it('applies profit/loss colors to bot P&L', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        const profitElement = screen.getByText(/\+\$1500\.00/);
        expect(profitElement).toHaveClass('text-profit');

        const lossElement = screen.getByText(/-\$300\.00/);
        expect(lossElement).toHaveClass('text-loss');
      });
    });
  });

  describe('Tax Report section', () => {
    it('renders tax summary correctly', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText(/Total Capital Gains \(2025\)/)).toBeInTheDocument();
        expect(screen.getByText(/\+\$2800\.00/)).toBeInTheDocument();
      });
    });

    it('displays tax year correctly', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Tax Year')).toBeInTheDocument();
        expect(screen.getByText('2025')).toBeInTheDocument();
      });
    });

    it('handles negative total gains', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            ...mockTaxReport,
            total_gains: -1500.0,
          }),
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText(/-\$1500\.00/)).toBeInTheDocument();
      });
    });

    it('renders tax entries table', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('BTC')).toBeInTheDocument();
        expect(screen.getByText('ETH')).toBeInTheDocument();
        expect(screen.getByText(/\$45000\.0000/)).toBeInTheDocument();
        expect(screen.getByText(/\$48000\.0000/)).toBeInTheDocument();
      });
    });

    it('formats dates correctly in tax table', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        // Check that dates are formatted (specific format may vary by locale)
        const dateElements = screen.getAllByText(/\/|,/);
        expect(dateElements.length).toBeGreaterThan(0);
      });
    });

    it('applies profit/loss colors to gain/loss column', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        const gainElement = screen.getByText(/\+\$3000\.00/);
        expect(gainElement).toHaveClass('text-profit');

        const lossElement = screen.getByText(/-\$200\.00/);
        expect(lossElement).toHaveClass('text-loss');
      });
    });

    it('handles empty tax entries', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            entries: [],
            total_gains: 0,
            year: 2025,
          }),
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText(/No taxable events recorded for 2025/i)).toBeInTheDocument();
      });
    });

    it('displays export CSV button', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Export CSV')).toBeInTheDocument();
      });
    });
  });

  describe('Fees Report section', () => {
    it('renders total fees correctly', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockFeeReport,
        });

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText('Total Fees Paid')).toBeInTheDocument();
        expect(screen.getByText(/\$40\.50/)).toBeInTheDocument();
      });
    });

    it('displays fees in loss color', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockFeeReport,
        });

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        const totalFeesElement = screen.getByText(/\$40\.50/);
        expect(totalFeesElement).toHaveClass('text-loss');
      });
    });

    it('renders fees by bot table', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockFeeReport,
        });

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText('Fees by Bot')).toBeInTheDocument();
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.getByText('Bot Beta')).toBeInTheDocument();
        expect(screen.getByText('50')).toBeInTheDocument();
        expect(screen.getByText('30')).toBeInTheDocument();
      });
    });

    it('handles empty fee entries', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            entries: [],
            total_fees: 0,
          }),
        });

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText(/No fee data recorded yet/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error handling', () => {
    it('displays error message when P&L API fails', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new Error('Failed to fetch P&L report')
      );

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/No P&L data available/i)).toBeInTheDocument();
      });
    });

    it('displays error message when tax API fails', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockRejectedValueOnce(new Error('Failed to fetch tax report'));

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText(/No tax data available/i)).toBeInTheDocument();
      });
    });

    it('displays error message when fees API fails', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockRejectedValueOnce(new Error('Failed to fetch fee report'));

      renderReports();

      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText(/No fee data available/i)).toBeInTheDocument();
      });
    });

    it('handles malformed P&L response gracefully', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ unexpected: 'data' }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Reports')).toBeInTheDocument();
      });
    });

    it('handles null data gracefully', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => null,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/No P&L data available/i)).toBeInTheDocument();
      });
    });
  });

  describe('Controls', () => {
    it('export button triggers download', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

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

    it('tab switching updates active tab', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const pnlTab = screen.getByText('P&L Report');
      const taxTab = screen.getByText('Tax Export');

      // Initially P&L tab is active
      expect(pnlTab.closest('button')).toHaveClass('border-accent');

      // Click tax tab
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(taxTab.closest('button')).toHaveClass('border-accent');
      });
    });

    it('switching tabs changes visible content', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockFeeReport,
        });

      renderReports();

      // P&L content visible
      await waitFor(() => {
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });

      // Switch to tax
      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Total Capital Gains (2025)')).toBeInTheDocument();
      });

      // Switch to fees
      const feesTab = screen.getByText('Fees');
      fireEvent.click(feesTab);

      await waitFor(() => {
        expect(screen.getByText('Fees by Bot')).toBeInTheDocument();
      });
    });

    it('can switch back to P&L tab', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('Tax Year')).toBeInTheDocument();
      });

      const pnlTab = screen.getByText('P&L Report');
      fireEvent.click(pnlTab);

      await waitFor(() => {
        expect(screen.getByText('P&L by Bot')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has main heading', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        const heading = screen.getByText('Reports');
        expect(heading.tagName).toBe('H2');
      });
    });

    it('has section heading for P&L table', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        const heading = screen.getByText('P&L by Bot');
        expect(heading.tagName).toBe('H3');
      });
    });

    it('table headers exist', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Bot')).toBeInTheDocument();
        expect(screen.getByText('Pair')).toBeInTheDocument();
        expect(screen.getByText('Strategy')).toBeInTheDocument();
        expect(screen.getByText('Win Rate')).toBeInTheDocument();
        expect(screen.getByText('Fees')).toBeInTheDocument();
        expect(screen.getByText('P&L')).toBeInTheDocument();
      });
    });

    it('buttons are keyboard accessible', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const tabs = screen.getAllByRole('button');
      expect(tabs.length).toBeGreaterThan(0);

      tabs.forEach(tab => {
        expect(tab.tagName).toBe('BUTTON');
      });
    });

    it('tabs have descriptive text', () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      expect(screen.getByText('P&L Report')).toBeInTheDocument();
      expect(screen.getByText('Tax Export')).toBeInTheDocument();
      expect(screen.getByText('Fees')).toBeInTheDocument();
    });

    it('export button has descriptive text', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        const exportButton = screen.getByText('Export CSV');
        expect(exportButton).toBeInTheDocument();
        expect(exportButton.closest('button')).toBeTruthy();
      });
    });

    it('empty states have helpful messages', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          entries: [],
          total_pnl: 0,
          overall_win_rate: 0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/No bots with P&L data yet/i)).toBeInTheDocument();
      });
    });
  });

  describe('Data formatting', () => {
    it('formats P&L with two decimals', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/\+\$1200\.00/)).toBeInTheDocument();
        expect(screen.getByText(/\+\$1500\.00/)).toBeInTheDocument();
      });
    });

    it('formats win rate with one decimal', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('66.7%')).toBeInTheDocument();
        expect(screen.getByText('33.3%')).toBeInTheDocument();
      });
    });

    it('formats prices with four decimals in tax report', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockTaxReport,
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText(/\$45000\.0000/)).toBeInTheDocument();
        expect(screen.getByText(/\$48000\.0000/)).toBeInTheDocument();
      });
    });

    it('replaces underscores in strategy names', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockPnLReport,
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('mean reversion')).toBeInTheDocument();
      });
    });
  });

  describe('Edge cases', () => {
    it('handles zero P&L', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          ...mockPnLReport,
          total_pnl: 0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/\+\$0\.00/)).toBeInTheDocument();
      });
    });

    it('handles very large P&L values', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          ...mockPnLReport,
          total_pnl: 999999.99,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText(/\+\$999999\.99/)).toBeInTheDocument();
      });
    });

    it('handles 100% win rate', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          ...mockPnLReport,
          overall_win_rate: 100.0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('100.0%')).toBeInTheDocument();
      });
    });

    it('handles 0% win rate', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          ...mockPnLReport,
          overall_win_rate: 0.0,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('0.0%')).toBeInTheDocument();
      });
    });

    it('handles single bot in report', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({
          entries: [mockPnLReport.entries[0]],
          total_pnl: 1500.0,
          overall_win_rate: 66.7,
        }),
      });

      renderReports();

      await waitFor(() => {
        expect(screen.getByText('Bot Alpha')).toBeInTheDocument();
        expect(screen.queryByText('Bot Beta')).not.toBeInTheDocument();
      });
    });

    it('handles single tax entry', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockPnLReport,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            entries: [mockTaxReport.entries[0]],
            total_gains: 3000.0,
            year: 2025,
          }),
        });

      renderReports();

      const taxTab = screen.getByText('Tax Export');
      fireEvent.click(taxTab);

      await waitFor(() => {
        expect(screen.getByText('BTC')).toBeInTheDocument();
        expect(screen.queryByText('ETH')).not.toBeInTheDocument();
      });
    });
  });
});
