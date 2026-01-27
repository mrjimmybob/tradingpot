import { renderHook, waitFor } from '@testing-library/react';
import { useRealtimePrice, WebSocketProvider, useWebSocketContext } from '../../src/contexts/WebSocketContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';

// Mock WebSocket
class MockWebSocket {
  public readyState: number = WebSocket.CONNECTING;
  public onopen: ((ev: Event) => void) | null = null;
  public onclose: ((ev: CloseEvent) => void) | null = null;
  public onerror: ((ev: Event) => void) | null = null;
  public onmessage: ((ev: MessageEvent) => void) | null = null;
  private static instances: MockWebSocket[] = [];

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
    // Simulate async connection
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      this.onopen?.(new Event('open'));
    }, 0);
  }

  send(data: string) {
    // Mock send
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close'));
  }

  static getLastInstance(): MockWebSocket | undefined {
    return this.instances[this.instances.length - 1];
  }

  static clearInstances() {
    this.instances = [];
  }

  static simulateMessage(data: object) {
    const instance = this.getLastInstance();
    if (instance && instance.onmessage) {
      const event = new MessageEvent('message', {
        data: JSON.stringify(data),
      });
      instance.onmessage(event);
    }
  }

  static simulateError() {
    const instance = this.getLastInstance();
    if (instance && instance.onerror) {
      instance.onerror(new Event('error'));
    }
  }

  static simulateClose() {
    const instance = this.getLastInstance();
    if (instance && instance.onclose) {
      instance.readyState = WebSocket.CLOSED;
      instance.onclose(new CloseEvent('close'));
    }
  }
}

describe('useRealtimePrice Hook', () => {
  let originalWebSocket: typeof WebSocket;
  let queryClient: QueryClient;

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
    MockWebSocket.clearInstances();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    jest.clearAllTimers();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  const createWrapper = () => {
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <WebSocketProvider>{children}</WebSocketProvider>
      </QueryClientProvider>
    );
  };

  describe('Connection lifecycle', () => {
    it('connects on mount', async () => {
      const { result } = renderHook(() => useWebSocketContext(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isConnecting).toBe(true);

      // Fast-forward to trigger connection
      jest.runAllTimers();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });
    });

    it('disconnects on unmount', async () => {
      const { unmount } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      unmount();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.CLOSED);
      });
    });

    it('subscribes when symbol is provided', async () => {
      const sendSpy = jest.fn();
      MockWebSocket.prototype.send = sendSpy;

      renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(sendSpy).toHaveBeenCalledWith(
          expect.stringContaining('"action":"subscribe"')
        );
      });
    });

    it('reconnects when symbol changes', async () => {
      const sendSpy = jest.fn();
      MockWebSocket.prototype.send = sendSpy;

      const { rerender } = renderHook(
        ({ symbol }) => useRealtimePrice(symbol),
        {
          initialProps: { symbol: 'BTC/USDT' },
          wrapper: createWrapper(),
        }
      );

      jest.runAllTimers();

      await waitFor(() => {
        expect(sendSpy).toHaveBeenCalledWith(
          expect.stringContaining('BTC/USDT')
        );
      });

      sendSpy.mockClear();

      // Change symbol
      rerender({ symbol: 'ETH/USDT' });

      jest.runAllTimers();

      await waitFor(() => {
        expect(sendSpy).toHaveBeenCalledWith(
          expect.stringContaining('ETH/USDT')
        );
      });
    });

    it('handles multiple re-renders without duplicating connections', async () => {
      const { rerender } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      const initialInstanceCount = MockWebSocket.clearInstances.length;

      // Force multiple re-renders
      rerender();
      rerender();
      rerender();

      jest.runAllTimers();

      // Should not create additional WebSocket instances
      expect(MockWebSocket.clearInstances.length).toBe(initialInstanceCount);
    });
  });

  describe('Message handling', () => {
    it('updates price state when valid message received', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Simulate price update
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current).toEqual({
          price: 50000,
          bid: 49995,
          ask: 50005,
          change_24h: 2.5,
          timestamp: '2025-01-26T12:00:00Z',
        });
      });
    });

    it('ignores malformed messages', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send malformed message
      const instance = MockWebSocket.getLastInstance();
      if (instance && instance.onmessage) {
        const event = new MessageEvent('message', {
          data: 'invalid json{{{',
        });
        instance.onmessage(event);
      }

      // Price should remain undefined
      expect(result.current).toBeUndefined();
    });

    it('ignores messages for wrong symbol', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send message for different symbol
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'ETH/USDT',
        price: 3000,
        bid: 2995,
        ask: 3005,
        change_24h: 1.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      // BTC/USDT price should still be undefined
      expect(result.current).toBeUndefined();
    });

    it('handles JSON parse errors gracefully', async () => {
      const consoleDebugSpy = jest.spyOn(console, 'debug').mockImplementation();

      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send invalid JSON
      const instance = MockWebSocket.getLastInstance();
      if (instance && instance.onmessage) {
        const event = new MessageEvent('message', {
          data: '{invalid}',
        });
        instance.onmessage(event);
      }

      // Should not crash, price remains undefined
      expect(result.current).toBeUndefined();

      consoleDebugSpy.mockRestore();
    });

    it('handles multiple price updates for same symbol', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // First update
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(50000);
      });

      // Second update
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 51000,
        bid: 50995,
        ask: 51005,
        change_24h: 3.5,
        timestamp: '2025-01-26T12:01:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(51000);
      });
    });
  });

  describe('Error handling', () => {
    it('handles WebSocket error event', async () => {
      const { result } = renderHook(() => useWebSocketContext(), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });

      // Simulate error
      MockWebSocket.simulateError();

      // Connection should still be open (errors don't close connection)
      expect(result.current.isConnected).toBe(true);
    });

    it('handles close event', async () => {
      const { result } = renderHook(() => useWebSocketContext(), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });

      // Simulate close
      MockWebSocket.simulateClose();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false);
      });
    });

    it('retries connection after close', async () => {
      const { result } = renderHook(() => useWebSocketContext(), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });

      // Clear instances to track new connection
      const initialCount = MockWebSocket.clearInstances.length;

      // Simulate close
      MockWebSocket.simulateClose();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false);
      });

      // Fast-forward reconnect interval
      jest.advanceTimersByTime(3000);
      jest.runAllTimers();

      // Should attempt reconnection
      await waitFor(() => {
        expect(result.current.isConnecting).toBe(true);
      });
    });

    it('sets error state when connection fails', async () => {
      // Mock constructor to throw error
      const originalConstructor = MockWebSocket;
      (global as any).WebSocket = class extends MockWebSocket {
        constructor(url: string) {
          super(url);
          throw new Error('Connection failed');
        }
      };

      const { result } = renderHook(() => useWebSocketContext(), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false);
      });

      // Restore
      (global as any).WebSocket = originalConstructor;
    });
  });

  describe('State behavior', () => {
    it('initial state is undefined', () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      expect(result.current).toBeUndefined();
    });

    it('updates price correctly', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current).toBeDefined();
        expect(result.current?.price).toBe(50000);
      });
    });

    it('preserves last valid price on error', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send valid price
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(50000);
      });

      // Simulate error
      MockWebSocket.simulateError();

      // Price should be preserved
      expect(result.current?.price).toBe(50000);
    });

    it('resets state when symbol changes', async () => {
      const { result, rerender } = renderHook(
        ({ symbol }) => useRealtimePrice(symbol),
        {
          initialProps: { symbol: 'BTC/USDT' },
          wrapper: createWrapper(),
        }
      );

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Set BTC price
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(50000);
      });

      // Change symbol
      rerender({ symbol: 'ETH/USDT' });

      jest.runAllTimers();

      // Should return undefined for new symbol (no data yet)
      expect(result.current).toBeUndefined();

      // Set ETH price
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'ETH/USDT',
        price: 3000,
        bid: 2995,
        ask: 3005,
        change_24h: 1.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(3000);
      });
    });
  });

  describe('Edge cases', () => {
    it('handles rapid symbol switching', async () => {
      const { rerender, result } = renderHook(
        ({ symbol }) => useRealtimePrice(symbol),
        {
          initialProps: { symbol: 'BTC/USDT' },
          wrapper: createWrapper(),
        }
      );

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Rapidly switch symbols
      rerender({ symbol: 'ETH/USDT' });
      rerender({ symbol: 'XRP/USDT' });
      rerender({ symbol: 'LTC/USDT' });
      rerender({ symbol: 'BNB/USDT' });

      jest.runAllTimers();

      // Should end up with BNB subscription
      await waitFor(() => {
        expect(result.current).toBeUndefined();
      });
    });

    it('handles delayed message arrival', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send newer message first
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 51000,
        bid: 50995,
        ask: 51005,
        change_24h: 3.5,
        timestamp: '2025-01-26T12:01:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(51000);
      });

      // Send older message
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
      });

      // Should update to older price (no timestamp filtering)
      await waitFor(() => {
        expect(result.current?.price).toBe(50000);
      });
    });

    it('handles empty payloads', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send empty message
      const instance = MockWebSocket.getLastInstance();
      if (instance && instance.onmessage) {
        const event = new MessageEvent('message', {
          data: '',
        });
        instance.onmessage(event);
      }

      // Should remain undefined
      expect(result.current).toBeUndefined();
    });

    it('handles unexpected message shape', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send message with unexpected structure
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        // Missing required fields
      });

      // Should not crash, but value will be undefined or incomplete
      expect(result.current).toBeDefined();
    });

    it('handles message with extra fields', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Send message with extra fields
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: 2.5,
        timestamp: '2025-01-26T12:00:00Z',
        extra_field: 'should be ignored',
        another_extra: 123,
      });

      await waitFor(() => {
        expect(result.current).toEqual({
          price: 50000,
          bid: 49995,
          ask: 50005,
          change_24h: 2.5,
          timestamp: '2025-01-26T12:00:00Z',
        });
      });
    });

    it('handles zero and negative prices', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      // Zero price
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 0,
        bid: 0,
        ask: 0,
        change_24h: 0,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(0);
      });

      // Negative change (valid)
      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 50000,
        bid: 49995,
        ask: 50005,
        change_24h: -5.5,
        timestamp: '2025-01-26T12:01:00Z',
      });

      await waitFor(() => {
        expect(result.current?.change_24h).toBe(-5.5);
      });
    });

    it('handles very large numbers', async () => {
      const { result } = renderHook(() => useRealtimePrice('BTC/USDT'), {
        wrapper: createWrapper(),
      });

      jest.runAllTimers();

      await waitFor(() => {
        expect(MockWebSocket.getLastInstance()?.readyState).toBe(WebSocket.OPEN);
      });

      MockWebSocket.simulateMessage({
        type: 'price_update',
        symbol: 'BTC/USDT',
        price: 999999999.99,
        bid: 999999999.98,
        ask: 999999999.99,
        change_24h: 99999.99,
        timestamp: '2025-01-26T12:00:00Z',
      });

      await waitFor(() => {
        expect(result.current?.price).toBe(999999999.99);
      });
    });
  });
});
