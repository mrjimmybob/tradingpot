/**
 * Frontend API Client Tests
 * 
 * Tests API client functions that wrap fetch calls to the backend.
 * These tests validate request formatting, response handling, and error cases.
 */

// Mock API client functions (extracted from pages for testing)
async function fetchStats(): Promise<any> {
  const res = await fetch('/api/stats');
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

async function fetchBots(status?: string): Promise<any[]> {
  const url = status ? `/api/bots?status=${status}` : '/api/bots';
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch bots');
  return res.json();
}

async function fetchBot(id: string): Promise<any> {
  const res = await fetch(`/api/bots/${id}`);
  if (!res.ok) {
    if (res.status === 404) throw new Error('Bot not found');
    throw new Error('Failed to fetch bot');
  }
  return res.json();
}

async function createBot(data: any): Promise<any> {
  const res = await fetch('/api/bots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create bot');
  return res.json();
}

async function updateBot(id: string, data: any): Promise<any> {
  const res = await fetch(`/api/bots/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update bot');
  return res.json();
}

async function deleteBot(id: string): Promise<void> {
  const res = await fetch(`/api/bots/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete bot');
}

async function startBot(id: string): Promise<any> {
  const res = await fetch(`/api/bots/${id}/start`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start bot');
  return res.json();
}

async function fetchTaxSummary(year: number, isSimulated: boolean): Promise<any> {
  const res = await fetch(
    `/api/reports/tax-summary?year=${year}&is_simulated=${isSimulated}`
  );
  if (!res.ok) throw new Error('Failed to fetch tax summary');
  return res.json();
}

async function fetchAuditLog(params: {
  bot_id?: number;
  severity?: string;
  start_date?: string;
  end_date?: string;
}): Promise<any[]> {
  const queryParams = new URLSearchParams();
  if (params.bot_id) queryParams.set('bot_id', params.bot_id.toString());
  if (params.severity) queryParams.set('severity', params.severity);
  if (params.start_date) queryParams.set('start_date', params.start_date);
  if (params.end_date) queryParams.set('end_date', params.end_date);

  const url = `/api/reports/audit?${queryParams.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch audit log');
  return res.json();
}

async function updateDataSource(sourceType: string, config: any): Promise<any> {
  const res = await fetch(`/api/data-sources/sources/${sourceType}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to update data source');
  return res.json();
}

async function bulkUpdateDataSources(updates: any): Promise<any> {
  const res = await fetch('/api/data-sources/sources/bulk', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error('Failed to bulk update data sources');
  return res.json();
}

describe('API Client Tests', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Request behavior', () => {
    it('uses GET method for fetch operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 5 }),
      });

      await fetchStats();

      expect(global.fetch).toHaveBeenCalledWith('/api/stats');
    });

    it('uses POST method for create operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      const botData = { name: 'Test Bot', trading_pair: 'BTC/USDT' };
      await createBot(botData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('uses PUT method for update operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      const updateData = { name: 'Updated Bot' };
      await updateBot('1', updateData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots/1',
        expect.objectContaining({ method: 'PUT' })
      );
    });

    it('uses DELETE method for delete operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
      });

      await deleteBot('1');

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots/1',
        expect.objectContaining({ method: 'DELETE' })
      );
    });

    it('constructs correct endpoint URL with path parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 123 }),
      });

      await fetchBot('123');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots/123');
    });

    it('constructs correct query parameters for filtering', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      await fetchBots('running');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots?status=running');
    });

    it('constructs multiple query parameters correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ year: 2025 }),
      });

      await fetchTaxSummary(2025, true);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/reports/tax-summary?year=2025&is_simulated=true'
      );
    });

    it('sends correct request body for POST', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      const botData = { name: 'Test Bot', trading_pair: 'BTC/USDT' };
      await createBot(botData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          body: JSON.stringify(botData),
        })
      );
    });

    it('sends correct request body for PUT', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      const updateData = { name: 'Updated Bot' };
      await updateBot('1', updateData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots/1',
        expect.objectContaining({
          body: JSON.stringify(updateData),
        })
      );
    });

    it('sets Content-Type header for JSON requests', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      await createBot({ name: 'Test' });

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('builds complex query strings correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      await fetchAuditLog({
        bot_id: 5,
        severity: 'error',
        start_date: '2025-01-01',
        end_date: '2025-12-31',
      });

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/reports/audit?bot_id=5&severity=error&start_date=2025-01-01&end_date=2025-12-31'
      );
    });

    it('omits optional query parameters when not provided', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      await fetchAuditLog({ severity: 'warning' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('severity=warning')
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.not.stringContaining('bot_id')
      );
    });
  });

  describe('Response handling', () => {
    it('parses successful JSON response', async () => {
      const mockData = { total_bots: 5, running_bots: 3 };
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockData,
      });

      const result = await fetchStats();

      expect(result).toEqual(mockData);
    });

    it('returns array for list endpoints', async () => {
      const mockBots = [
        { id: 1, name: 'Bot 1' },
        { id: 2, name: 'Bot 2' },
      ];
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => mockBots,
      });

      const result = await fetchBots();

      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);
    });

    it('handles empty array response', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      const result = await fetchBots();

      expect(result).toEqual([]);
    });

    it('handles empty object response', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      const result = await fetchStats();

      expect(result).toEqual({});
    });

    it('handles response with null values', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 0, running_bots: null }),
      });

      const result = await fetchStats();

      expect(result.total_bots).toBe(0);
      expect(result.running_bots).toBeNull();
    });

    it('handles malformed JSON response', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => {
          throw new SyntaxError('Unexpected token');
        },
      });

      await expect(fetchStats()).rejects.toThrow();
    });

    it('handles unexpected response shape', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => 'not an object',
      });

      const result = await fetchStats();

      expect(result).toBe('not an object');
    });

    it('returns void for DELETE operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
      });

      const result = await deleteBot('1');

      expect(result).toBeUndefined();
    });
  });

  describe('Error handling', () => {
    it('throws error on 400 Bad Request', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 400,
      });

      await expect(fetchStats()).rejects.toThrow('Failed to fetch stats');
    });

    it('throws error on 401 Unauthorized', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 401,
      });

      await expect(fetchStats()).rejects.toThrow('Failed to fetch stats');
    });

    it('throws specific error on 404 Not Found', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 404,
      });

      await expect(fetchBot('999')).rejects.toThrow('Bot not found');
    });

    it('throws error on 500 Internal Server Error', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
      });

      await expect(fetchStats()).rejects.toThrow('Failed to fetch stats');
    });

    it('throws error on network failure', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new Error('Network error')
      );

      await expect(fetchStats()).rejects.toThrow('Network error');
    });

    it('throws error when fetch rejects', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new TypeError('Failed to fetch')
      );

      await expect(fetchStats()).rejects.toThrow('Failed to fetch');
    });

    it('propagates error for POST failures', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 400,
      });

      await expect(createBot({ name: 'Test' })).rejects.toThrow(
        'Failed to create bot'
      );
    });

    it('propagates error for PUT failures', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 403,
      });

      await expect(updateBot('1', { name: 'Test' })).rejects.toThrow(
        'Failed to update bot'
      );
    });

    it('propagates error for DELETE failures', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 404,
      });

      await expect(deleteBot('1')).rejects.toThrow('Failed to delete bot');
    });

    it('handles timeout errors', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new Error('Request timeout')
      );

      await expect(fetchStats()).rejects.toThrow('Request timeout');
    });

    it('handles CORS errors', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new TypeError('CORS policy blocked')
      );

      await expect(fetchStats()).rejects.toThrow('CORS policy blocked');
    });
  });

  describe('Edge cases', () => {
    it('handles missing required parameters gracefully', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      // Call with no parameters
      await fetchBots();

      expect(global.fetch).toHaveBeenCalledWith('/api/bots');
    });

    it('handles null input', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      await createBot(null);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          body: 'null',
        })
      );
    });

    it('handles undefined input', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      await createBot(undefined);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          body: undefined,
        })
      );
    });

    it('handles empty object input', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      await createBot({});

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          body: '{}',
        })
      );
    });

    it('handles empty string parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      await fetchBots('');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots?status=');
    });

    it('handles special characters in parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      await fetchBot('bot-123%20test');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots/bot-123%20test');
    });

    it('handles very large response payloads', async () => {
      const largeArray = Array.from({ length: 10000 }, (_, i) => ({
        id: i,
        name: `Bot ${i}`,
      }));
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => largeArray,
      });

      const result = await fetchBots();

      expect(result).toHaveLength(10000);
    });

    it('handles rapid successive calls', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 5 }),
      });

      await Promise.all([
        fetchStats(),
        fetchStats(),
        fetchStats(),
      ]);

      expect(global.fetch).toHaveBeenCalledTimes(3);
    });

    it('handles concurrent requests to different endpoints', async () => {
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        return Promise.resolve({
          ok: true,
          json: async () => (url.includes('stats') ? { total_bots: 5 } : []),
        });
      });

      await Promise.all([
        fetchStats(),
        fetchBots(),
      ]);

      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('handles nested objects in request body', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ id: 1 }),
      });

      const nestedData = {
        name: 'Test Bot',
        config: {
          risk: { max_loss: 5.0 },
          strategy: { name: 'momentum' },
        },
      };

      await createBot(nestedData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/bots',
        expect.objectContaining({
          body: JSON.stringify(nestedData),
        })
      );
    });

    it('handles arrays in request body', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ success: true }),
      });

      const bulkData = {
        updates: [
          { id: 1, name: 'Bot 1' },
          { id: 2, name: 'Bot 2' },
        ],
      };

      await bulkUpdateDataSources(bulkData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/data-sources/sources/bulk',
        expect.objectContaining({
          body: JSON.stringify(bulkData),
        })
      );
    });

    it('is idempotent for GET requests', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 5 }),
      });

      const result1 = await fetchStats();
      const result2 = await fetchStats();

      expect(result1).toEqual(result2);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('handles numeric zero values correctly', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 0, total_pnl: 0 }),
      });

      const result = await fetchStats();

      expect(result.total_bots).toBe(0);
      expect(result.total_pnl).toBe(0);
    });

    it('handles boolean values in query parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ year: 2025 }),
      });

      await fetchTaxSummary(2025, false);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('is_simulated=false')
      );
    });

    it('handles multiple simultaneous errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 500,
      });

      const errors = await Promise.allSettled([
        fetchStats(),
        fetchBots(),
        fetchBot('1'),
      ]);

      expect(errors.every((e) => e.status === 'rejected')).toBe(true);
    });

    it('calls fetch exactly once per request', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ total_bots: 5 }),
      });

      await fetchStats();

      expect(global.fetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('Bulk operations', () => {
    it('handles bulk update with multiple items', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ updated: 3 }),
      });

      const updates = {
        sources: [
          { type: 'source1', enabled: true },
          { type: 'source2', enabled: false },
          { type: 'source3', enabled: true },
        ],
      };

      await bulkUpdateDataSources(updates);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/data-sources/sources/bulk',
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(updates),
        })
      );
    });

    it('handles empty bulk operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ updated: 0 }),
      });

      await bulkUpdateDataSources({ sources: [] });

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/data-sources/sources/bulk',
        expect.objectContaining({
          body: JSON.stringify({ sources: [] }),
        })
      );
    });
  });

  describe('URL construction', () => {
    it('constructs URL without trailing slashes', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      await fetchBot('123');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots/123');
    });

    it('constructs URL with action suffix', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'running' }),
      });

      await startBot('5');

      expect(global.fetch).toHaveBeenCalledWith('/api/bots/5/start');
    });

    it('preserves URL encoding in parameters', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => [],
      });

      await fetchAuditLog({ severity: 'error warning' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('severity=error+warning')
      );
    });
  });
});
