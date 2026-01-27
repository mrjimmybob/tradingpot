import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AuditCompliance } from '../../src/pages/AuditCompliance';

const renderAuditCompliance = () => {
  return render(
    <MemoryRouter>
      <AuditCompliance />
    </MemoryRouter>
  );
};

describe('AuditCompliance Page', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders without crashing', () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    renderAuditCompliance();
    expect(screen.getByText(/Audit/i) || screen.getByText(/Compliance/i)).toBeInTheDocument();
  });

  it('renders audit log list', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'info',
        source: 'trading_engine',
        bot_id: 1,
        message: 'Trade executed successfully',
        details: null,
      },
      {
        id: 2,
        timestamp: '2025-01-15T11:00:00Z',
        severity: 'warning',
        source: 'risk_management',
        bot_id: 1,
        message: 'Stop loss triggered',
        details: { loss_pct: 5.0 },
      },
      {
        id: 3,
        timestamp: '2025-01-15T12:00:00Z',
        severity: 'error',
        source: 'exchange',
        bot_id: 2,
        message: 'Order rejected by exchange',
        details: { reason: 'Insufficient balance' },
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Trade executed successfully/i)).toBeInTheDocument();
      expect(screen.getByText(/Stop loss triggered/i)).toBeInTheDocument();
      expect(screen.getByText(/Order rejected by exchange/i)).toBeInTheDocument();
    });
  });

  it('supports severity filter', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'error',
        source: 'trading_engine',
        bot_id: 1,
        message: 'Critical error',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Critical error/i)).toBeInTheDocument();
    });

    const severityFilter = screen.getByRole('combobox', { name: /severity/i }) ||
                          screen.getAllByRole('combobox').find(el => 
                            el.textContent?.includes('Severity') || 
                            el.getAttribute('name')?.includes('severity')
                          );

    if (severityFilter) {
      fireEvent.change(severityFilter, { target: { value: 'error' } });

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('severity=error')
        );
      });
    }
  });

  it('supports date range filter', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });

    const startDateInput = screen.getByLabelText(/start date/i) ||
                          screen.getAllByRole('textbox').find(el => 
                            el.getAttribute('type') === 'date' ||
                            el.getAttribute('name')?.includes('start')
                          );

    if (startDateInput) {
      fireEvent.change(startDateInput, { target: { value: '2025-01-01' } });

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('start_date')
        );
      });
    }
  });

  it('shows empty state when no records', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(
        screen.getByText(/No audit logs/i) || 
        screen.getByText(/No records/i) ||
        screen.getByText(/empty/i)
      ).toBeInTheDocument();
    });
  });

  it('error handling when API fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Failed to fetch audit log'));

    renderAuditCompliance();

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

    renderAuditCompliance();

    expect(screen.getByText(/Loading/i) || screen.getByRole('status')).toBeTruthy();
  });

  it('displays severity icons correctly', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'error',
        source: 'system',
        bot_id: null,
        message: 'Error occurred',
        details: null,
      },
      {
        id: 2,
        timestamp: '2025-01-15T11:00:00Z',
        severity: 'warning',
        source: 'system',
        bot_id: null,
        message: 'Warning issued',
        details: null,
      },
      {
        id: 3,
        timestamp: '2025-01-15T12:00:00Z',
        severity: 'info',
        source: 'system',
        bot_id: null,
        message: 'Info message',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Error occurred/i)).toBeInTheDocument();
      expect(screen.getByText(/Warning issued/i)).toBeInTheDocument();
      expect(screen.getByText(/Info message/i)).toBeInTheDocument();
    });
  });

  it('filters by bot ID', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'info',
        source: 'trading_engine',
        bot_id: 5,
        message: 'Bot 5 trade',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Bot 5 trade/i)).toBeInTheDocument();
    });

    const botIdInput = screen.getByLabelText(/bot/i) ||
                      screen.getAllByRole('textbox').find(el => 
                        el.getAttribute('name')?.includes('bot')
                      );

    if (botIdInput) {
      fireEvent.change(botIdInput, { target: { value: '5' } });

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('bot_id=5')
        );
      });
    }
  });

  it('handles clear filters action', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    renderAuditCompliance();

    const clearButton = screen.getByRole('button', { name: /clear/i }) ||
                       screen.getByText(/clear/i).closest('button');

    if (clearButton) {
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    }
  });

  it('supports CSV export', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'info',
        source: 'system',
        bot_id: 1,
        message: 'Test message',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = jest.fn();

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Test message/i)).toBeInTheDocument();
    });

    const exportButton = screen.getByRole('button', { name: /export/i }) ||
                        screen.getByText(/download/i).closest('button');

    if (exportButton) {
      fireEvent.click(exportButton);

      expect(global.URL.createObjectURL).toHaveBeenCalled();
    }
  });

  it('displays simulated vs live mode toggle', () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    renderAuditCompliance();

    expect(
      screen.getByText(/Simulated/i) ||
      screen.getByText(/Live/i) ||
      screen.getByRole('checkbox')
    ).toBeTruthy();
  });

  it('handles expandable log details', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'error',
        source: 'trading_engine',
        bot_id: 1,
        message: 'Error with details',
        details: { error_code: 'ERR_001', stack: 'Error stack trace' },
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/Error with details/i)).toBeInTheDocument();
    });

    const expandButton = screen.queryByText(/details/i) ||
                        screen.queryByRole('button', { name: /expand/i });

    if (expandButton) {
      fireEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText(/ERR_001/i) || screen.getByText(/stack/i)).toBeInTheDocument();
      });
    }
  });

  it('handles 500 error from server', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(
        screen.getByText(/Failed to fetch/i) || screen.getByText(/error/i)
      ).toBeInTheDocument();
    });
  });

  it('displays timestamp in readable format', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:30:45Z',
        severity: 'info',
        source: 'system',
        bot_id: null,
        message: 'Timestamp test',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(
        screen.getByText(/2025/) || 
        screen.getByText(/Jan/) ||
        screen.getByText(/10:30/)
      ).toBeTruthy();
    });
  });

  it('handles logs without bot_id (system-level)', async () => {
    const mockLogs = [
      {
        id: 1,
        timestamp: '2025-01-15T10:00:00Z',
        severity: 'info',
        source: 'system',
        bot_id: null,
        message: 'System-level event',
        details: null,
      },
    ];

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockLogs,
    });

    renderAuditCompliance();

    await waitFor(() => {
      expect(screen.getByText(/System-level event/i)).toBeInTheDocument();
    });
  });
});
