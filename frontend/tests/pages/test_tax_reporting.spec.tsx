import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { TaxReporting } from '../../src/pages/TaxReporting';

const renderTaxReporting = () => {
  return render(
    <MemoryRouter>
      <TaxReporting />
    </MemoryRouter>
  );
};

describe('TaxReporting Page', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders without crashing', () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 0,
        short_term_gain: 0,
        long_term_gain: 0,
        lot_count: 0,
        trade_count: 0,
      }),
    });

    renderTaxReporting();
    expect(screen.getByText(/Tax Reporting/i)).toBeInTheDocument();
  });

  it('renders tax summary table', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 5000.0,
        short_term_gain: 3000.0,
        long_term_gain: 2000.0,
        lot_count: 10,
        trade_count: 25,
      }),
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(screen.getByText(/\$5,000\.00/) || screen.getByText(/5000/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText(/\$3,000\.00/) || screen.getByText(/3000/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText(/\$2,000\.00/) || screen.getByText(/2000/)).toBeInTheDocument();
    });
  });

  it('loads data from API', async () => {
    const mockData = {
      total_realized_gain: 1234.56,
      short_term_gain: 800.0,
      long_term_gain: 434.56,
      lot_count: 5,
      trade_count: 15,
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/reports/tax-summary')
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/\$1,234\.56/) || screen.getByText(/1234\.56/)).toBeInTheDocument();
    });
  });

  it('handles year filter changes', async () => {
    const currentYear = new Date().getFullYear();

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 1000.0,
        short_term_gain: 600.0,
        long_term_gain: 400.0,
        lot_count: 3,
        trade_count: 10,
      }),
    });

    renderTaxReporting();

    const yearSelector = screen.getByRole('combobox', { name: /year/i }) || 
                        screen.getAllByRole('combobox')[0];

    fireEvent.change(yearSelector, { target: { value: String(currentYear - 1) } });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining(`tax-summary/${currentYear - 1}`)
      );
    });
  });

  it('shows zero-gains case correctly', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 0,
        short_term_gain: 0,
        long_term_gain: 0,
        lot_count: 0,
        trade_count: 0,
      }),
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(screen.getByText(/\$0\.00/) || screen.getByText('0')).toBeInTheDocument();
    });
  });

  it('handles negative gains correctly', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: -500.0,
        short_term_gain: -300.0,
        long_term_gain: -200.0,
        lot_count: 5,
        trade_count: 12,
      }),
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(
        screen.getByText(/-\$500\.00/) ||
        screen.getByText(/-500/) ||
        screen.getByText(/\(\$500\.00\)/)
      ).toBeInTheDocument();
    });
  });

  it('error handling when API fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Failed to fetch tax summary'));

    renderTaxReporting();

    await waitFor(() => {
      expect(
        screen.getByText(/Failed to fetch/i) ||
        screen.getByText(/error/i) ||
        screen.getByRole('alert')
      ).toBeInTheDocument();
    });
  });

  it('shows loading state while fetching', () => {
    (global.fetch as jest.Mock).mockImplementation(
      () =>
        new Promise(() => {
          // Never resolves
        })
    );

    renderTaxReporting();

    expect(screen.getByText(/Loading/i) || screen.getByRole('status')).toBeTruthy();
  });

  it('displays simulated vs live mode toggle', () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 0,
        short_term_gain: 0,
        long_term_gain: 0,
        lot_count: 0,
        trade_count: 0,
      }),
    });

    renderTaxReporting();

    expect(
      screen.getByText(/Simulated/i) || 
      screen.getByText(/Live/i) ||
      screen.getByRole('checkbox')
    ).toBeTruthy();
  });

  it('handles export CSV functionality', async () => {
    const mockBlob = new Blob(['csv data'], { type: 'text/csv' });

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          total_realized_gain: 1000.0,
          short_term_gain: 600.0,
          long_term_gain: 400.0,
          lot_count: 5,
          trade_count: 15,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        blob: async () => mockBlob,
      });

    renderTaxReporting();

    await waitFor(() => {
      expect(screen.getByText(/\$1,000\.00/)).toBeInTheDocument();
    });

    const exportButton = screen.getByRole('button', { name: /export/i }) ||
                        screen.getByText(/download/i).closest('button');

    if (exportButton) {
      fireEvent.click(exportButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/reports/tax-export'),
          expect.objectContaining({ method: 'POST' })
        );
      });
    }
  });

  it('shows lot count and trade count', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 2000.0,
        short_term_gain: 1200.0,
        long_term_gain: 800.0,
        lot_count: 15,
        trade_count: 42,
      }),
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(screen.getByText(/15/) && screen.getByText(/42/)).toBeTruthy();
    });
  });

  it('handles 500 error from server', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(
        screen.getByText(/Failed to fetch/i) || screen.getByText(/error/i)
      ).toBeInTheDocument();
    });
  });

  it('handles empty response from API', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => null,
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(
        screen.getByText(/no data/i) || 
        screen.getByText(/error/i) ||
        screen.queryByText(/\$/i)
      ).toBeTruthy();
    });
  });

  it('correctly formats large numbers', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_realized_gain: 123456.78,
        short_term_gain: 80000.0,
        long_term_gain: 43456.78,
        lot_count: 100,
        trade_count: 250,
      }),
    });

    renderTaxReporting();

    await waitFor(() => {
      expect(
        screen.getByText(/\$123,456\.78/) || screen.getByText(/123456\.78/)
      ).toBeInTheDocument();
    });
  });
});
