import { io } from 'socket.io-client'

const API_BASE = '/api'

// Socket.IO connection
let socket = null

export const connectSocket = (onProgress, onConnect, onDisconnect) => {
  socket = io('http://localhost:5000', {
    transports: ['websocket', 'polling'],
  })

  socket.on('connect', () => {
    console.log('WebSocket connected')
    onConnect?.()
  })

  socket.on('disconnect', () => {
    console.log('WebSocket disconnected')
    onDisconnect?.()
  })

  socket.on('progress', (data) => {
    console.log('Progress update:', data)
    onProgress?.(data)
  })

  socket.on('connected', (data) => {
    console.log('Initial status:', data)
    onProgress?.(data)
  })

  return socket
}

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect()
    socket = null
  }
}

export const requestStatus = () => {
  if (socket) {
    socket.emit('request_status')
  }
}

// REST API calls
export const api = {
  // Health check
  async health() {
    const res = await fetch(`${API_BASE}/health`)
    return res.json()
  },

  // Get configuration
  async getConfig() {
    const res = await fetch(`${API_BASE}/config`)
    return res.json()
  },

  // Get experiment status
  async getStatus() {
    const res = await fetch(`${API_BASE}/experiment/status`)
    return res.json()
  },

  // Start experiment
  async startExperiment(params = {}) {
    const res = await fetch(`${API_BASE}/experiment/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })
    return res.json()
  },

  // Get results
  async getResults() {
    const res = await fetch(`${API_BASE}/experiment/results`)
    if (!res.ok) return null
    return res.json()
  },

  // Export CSV
  async exportCSV() {
    const res = await fetch(`${API_BASE}/export/csv`)
    if (!res.ok) throw new Error('Export failed')
    const blob = await res.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'experiment_results.csv'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  },

  // Export JSON
  async exportJSON() {
    const results = await this.getResults()
    if (!results) throw new Error('No results to export')
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'experiment_results.json'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  },
}

export default api
