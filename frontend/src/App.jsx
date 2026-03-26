import React, { useState, useEffect, useCallback, useRef } from 'react'
import { 
  Activity, 
  Play, 
  BarChart3, 
  Download, 
  Moon, 
  Sun, 
  Loader2,
  CheckCircle2,
  AlertCircle,
  Waves,
  Settings,
  TrendingUp,
  Zap,
  Sparkles,
  Music,
  Network,
  ArrowRight,
  ChevronRight,
  Brain,
  Gauge,
  Timer,
  FileAudio,
  Layers,
  Radio,
  Headphones,
  Mic2,
  ScanSearch,
  ShieldCheck,
  GitBranch,
  BarChart2,
  Volume2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Switch } from '@/components/ui/switch'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts'
import { api, connectSocket, disconnectSocket } from '@/services/api'

// Color palette
const COLORS = ['#667eea', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316']
const GRADIENT_COLORS = {
  primary: ['#667eea', '#764ba2'],
  success: ['#11998e', '#38ef7d'],
  warning: ['#f093fb', '#f5576c'],
  info: ['#4facfe', '#00f2fe']
}

// Animated Number Counter
function AnimatedNumber({ value, duration = 1000, decimals = 0, suffix = '' }) {
  const [displayValue, setDisplayValue] = useState(0)
  const startTime = useRef(null)
  const startValue = useRef(0)

  useEffect(() => {
    startValue.current = displayValue
    startTime.current = Date.now()
    
    const animate = () => {
      const now = Date.now()
      const progress = Math.min((now - startTime.current) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
      const current = startValue.current + (value - startValue.current) * eased
      setDisplayValue(current)
      
      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }
    
    requestAnimationFrame(animate)
  }, [value, duration])

  return <span className="counter-value">{displayValue.toFixed(decimals)}{suffix}</span>
}

// Waveform Animation Component
function WaveformAnimation({ active = true }) {
  return (
    <div className="flex items-center justify-center h-8 gap-1">
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className={`w-1 bg-gradient-to-t from-primary to-purple-500 rounded-full transition-all duration-300 ${
            active ? 'animate-pulse' : ''
          }`}
          style={{
            height: active ? `${Math.random() * 20 + 10}px` : '4px',
            animationDelay: `${i * 0.1}s`
          }}
        />
      ))}
    </div>
  )
}

// Floating Particles Background
function ParticleBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      <div className="absolute top-20 left-10 w-72 h-72 bg-purple-500/10 rounded-full filter blur-3xl animate-blob" />
      <div className="absolute top-40 right-10 w-96 h-96 bg-blue-500/10 rounded-full filter blur-3xl animate-blob animation-delay-200" />
      <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-pink-500/10 rounded-full filter blur-3xl animate-blob animation-delay-400" />
    </div>
  )
}

// Glowing Stat Card
function GlowingStatCard({ icon: Icon, title, value, subtitle, color = 'blue', delay = 0 }) {
  const colorClasses = {
    blue: 'from-blue-500 to-cyan-500',
    green: 'from-emerald-500 to-green-500',
    purple: 'from-purple-500 to-pink-500',
    orange: 'from-orange-500 to-yellow-500'
  }

  return (
    <div 
      className="animate-fade-in-up opacity-0"
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <Card className="relative overflow-hidden card-hover glass-card border-0 group">
        <div className={`absolute inset-0 bg-gradient-to-br ${colorClasses[color]} opacity-0 group-hover:opacity-10 transition-opacity duration-500`} />
        <div className={`absolute -top-10 -right-10 w-32 h-32 bg-gradient-to-br ${colorClasses[color]} rounded-full opacity-10 blur-2xl group-hover:opacity-20 transition-opacity`} />
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
          <div className={`p-2 rounded-xl bg-gradient-to-br ${colorClasses[color]} shadow-lg`}>
            <Icon className="h-4 w-4 text-white" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold tracking-tight">{value}</div>
          <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        </CardContent>
      </Card>
    </div>
  )
}

// Progress Step Indicator
function ProgressSteps({ currentStep, steps }) {
  return (
    <div className="flex items-center justify-between w-full">
      {steps.map((step, index) => {
        const isComplete = index < currentStep
        const isCurrent = index === currentStep
        
        return (
          <React.Fragment key={step.key}>
            <div className="flex flex-col items-center">
              <div className={`
                relative w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500
                ${isComplete ? 'bg-gradient-to-br from-green-400 to-emerald-600 shadow-lg shadow-green-500/30' : 
                  isCurrent ? 'bg-gradient-to-br from-blue-400 to-purple-600 shadow-lg shadow-blue-500/30 animate-pulse' : 
                  'bg-muted'}
              `}>
                {isComplete ? (
                  <CheckCircle2 className="w-5 h-5 text-white" />
                ) : (
                  <span className={`text-sm font-semibold ${isCurrent ? 'text-white' : 'text-muted-foreground'}`}>
                    {index + 1}
                  </span>
                )}
                {isCurrent && (
                  <div className="absolute inset-0 rounded-full border-2 border-blue-400 animate-ping opacity-20" />
                )}
              </div>
              <span className={`mt-2 text-xs font-medium ${isCurrent ? 'text-primary' : 'text-muted-foreground'}`}>
                {step.label}
              </span>
            </div>
            {index < steps.length - 1 && (
              <div className="flex-1 h-1 mx-2 rounded-full bg-muted overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r from-green-400 to-emerald-600 transition-all duration-500 ${
                    isComplete ? 'w-full' : isCurrent ? 'w-1/2 animate-pulse' : 'w-0'
                  }`}
                />
              </div>
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

// Feature Card with animated icon
function FeatureCard({ icon: Icon, title, items, gradient }) {
  return (
    <div className="p-6 rounded-2xl bg-gradient-to-br from-background to-muted/50 border border-border/50 hover:border-primary/30 transition-all duration-300 group hover-lift">
      <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${gradient} mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
        <Icon className="h-6 w-6 text-white" />
      </div>
      <h4 className="font-semibold text-lg mb-3">{title}</h4>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-center gap-2 text-sm text-muted-foreground">
            <ChevronRight className="h-3 w-3 text-primary" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}

// Method Badge
function MethodBadge({ id, name, isNew }) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-gradient-to-r from-muted/50 to-muted/30 hover:from-primary/10 hover:to-purple-500/10 transition-all duration-300 group cursor-default">
      <div 
        className="w-3 h-3 rounded-full shadow-lg"
        style={{ 
          backgroundColor: COLORS[parseInt(id)],
          boxShadow: `0 0 10px ${COLORS[parseInt(id)]}50`
        }}
      />
      <span className="text-sm font-medium capitalize flex-1">{name.replace(/_/g, ' ')}</span>
      {isNew && (
        <span className="px-2 py-0.5 text-[10px] font-bold uppercase bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full animate-pulse">
          New
        </span>
      )}
    </div>
  )
}

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'true'
  })
  const [activeTab, setActiveTab] = useState('dashboard')
  const [connected, setConnected] = useState(false)
  const [config, setConfig] = useState(null)
  const [experimentStatus, setExperimentStatus] = useState({
    status: 'idle',
    progress: 0,
    message: ''
  })
  const [results, setResults] = useState(null)
  const [logs, setLogs] = useState([])
  const [experimentParams, setExperimentParams] = useState({
    max_samples: 1000,
    epochs: 20,
    routing_samples: 500,
    test_noise_snr: 0
  })

  // Dark mode effect
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('darkMode', darkMode)
  }, [darkMode])

  // Load config on mount
  useEffect(() => {
    api.getConfig().then(setConfig).catch(console.error)
  }, [])

  // WebSocket connection
  useEffect(() => {
    const handleProgress = (data) => {
      setExperimentStatus(data)
      if (data.message) {
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message: data.message }].slice(-100))
      }
      if (data.status === 'complete') {
        api.getResults().then(setResults)
      }
    }

    const socket = connectSocket(
      handleProgress,
      () => setConnected(true),
      () => setConnected(false)
    )

    return () => disconnectSocket()
  }, [])

  // Load existing results
  useEffect(() => {
    api.getResults().then(r => r && setResults(r)).catch(() => {})
  }, [])

  const startExperiment = async () => {
    setLogs([])
    setResults(null)
    try {
      await api.startExperiment(experimentParams)
    } catch (err) {
      console.error('Failed to start experiment:', err)
    }
  }

  const isRunning = ['loading', 'analyzing', 'training_baseline', 'training_routing', 'training_adaptive'].includes(experimentStatus.status)
  
  const getCurrentStepIndex = () => {
    const steps = ['idle', 'loading', 'analyzing', 'training_baseline', 'training_routing', 'training_adaptive', 'complete']
    return steps.indexOf(experimentStatus.status)
  }

  const progressSteps = [
    { key: 'loading', label: 'Load Data' },
    { key: 'training_baseline', label: 'Baseline' },
    { key: 'training_routing', label: 'Routing' },
    { key: 'training_adaptive', label: 'Adaptive' },
    { key: 'complete', label: 'Complete' }
  ]

  return (
    <div className="min-h-screen bg-background transition-theme relative">
      <ParticleBackground />
      
      {/* Header */}
      <header className="border-b sticky top-0 z-50 bg-background/80 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl blur-lg opacity-50 animate-pulse" />
              <div className="relative p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Waves className="h-6 w-6 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                Adaptive Audio Preprocessing
              </h1>
              <p className="text-sm text-muted-foreground flex items-center gap-2">
                <Sparkles className="h-3 w-3" />
                Model-Based Intelligent Routing
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50">
              <div className={`h-2 w-2 rounded-full transition-all duration-300 ${
                connected ? 'bg-green-500 shadow-lg shadow-green-500/50' : 'bg-red-500'
              }`} />
              <span className="text-sm text-muted-foreground">{connected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center gap-3 p-2 rounded-full bg-muted/50">
              <Sun className={`h-4 w-4 transition-all duration-300 ${!darkMode ? 'text-yellow-500' : 'text-muted-foreground'}`} />
              <Switch checked={darkMode} onCheckedChange={setDarkMode} />
              <Moon className={`h-4 w-4 transition-all duration-300 ${darkMode ? 'text-blue-400' : 'text-muted-foreground'}`} />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-lg mx-auto grid-cols-3 p-1 bg-muted/50 backdrop-blur-sm rounded-2xl">
            {[
              { value: 'dashboard', icon: Activity, label: 'Dashboard' },
              { value: 'experiment', icon: Play, label: 'Experiment' },
              { value: 'results', icon: BarChart3, label: 'Results' }
            ].map((tab) => (
              <TabsTrigger 
                key={tab.value}
                value={tab.value} 
                className="gap-2 rounded-xl data-[state=active]:bg-background data-[state=active]:shadow-lg transition-all duration-300"
                disabled={tab.value === 'results' && !results}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-10">

            {/* ── Hero ─────────────────────────────────────────────────────── */}
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 p-8 md:p-14 text-white">
              <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-30" />
              <div className="relative z-10 grid md:grid-cols-2 gap-8 items-center">
                <div>
                  <div className="flex items-center gap-2 mb-5">
                    <span className="px-3 py-1 text-xs font-bold uppercase tracking-wider bg-white/20 rounded-full backdrop-blur-sm">
                      Final Year Project
                    </span>
                    <span className="px-3 py-1 text-xs font-bold uppercase tracking-wider bg-emerald-400/30 rounded-full backdrop-blur-sm">
                      Music Genre Classification
                    </span>
                  </div>
                  <h2 className="text-3xl md:text-4xl font-extrabold leading-tight mb-4">
                    Smarter Audio. <br />
                    <span className="text-white/80">Better Classification.</span>
                  </h2>
                  <p className="text-white/75 text-base leading-relaxed mb-8 max-w-md">
                    This system listens to each audio track, diagnoses its noise conditions, then automatically applies the most suitable denoising method — so a neural network can classify the genre more accurately, even in noisy environments.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <Button
                      size="lg"
                      className="bg-white text-purple-700 hover:bg-white/90 shadow-xl font-semibold btn-shine"
                      onClick={() => setActiveTab('experiment')}
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Run the System
                    </Button>
                    {results && (
                      <Button
                        size="lg"
                        variant="outline"
                        className="border-white/40 text-white hover:bg-white/10"
                        onClick={() => setActiveTab('results')}
                      >
                        <BarChart3 className="mr-2 h-4 w-4" />
                        View Results
                      </Button>
                    )}
                  </div>
                </div>
                {/* Animated KPI strip */}
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: 'Audio Samples', value: '8,000+', sub: 'FMA dataset tracks', icon: Music },
                    { label: 'Genre Classes', value: config?.genres?.length ? `${config.genres.length}` : '8', sub: 'Distinct music genres', icon: Headphones },
                    { label: 'Noise Conditions', value: '5', sub: 'Clean → Severe SNR', icon: Volume2 },
                    { label: 'Routing Methods', value: config?.preprocessing_methods ? Object.keys(config.preprocessing_methods).length : '5', sub: 'Denoising strategies', icon: GitBranch },
                  ].map(({ label, value, sub, icon: Icon }, i) => (
                    <div key={i} className="p-4 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/20 transition-all duration-300">
                      <Icon className="h-5 w-5 mb-2 text-white/70" />
                      <div className="text-2xl font-bold">{value}</div>
                      <div className="text-xs font-semibold text-white/90">{label}</div>
                      <div className="text-xs text-white/60 mt-0.5">{sub}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ── How It Works: Pipeline Flow ──────────────────────────────── */}
            <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 overflow-hidden">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500">
                    <GitBranch className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <CardTitle>How It Works</CardTitle>
                    <CardDescription>End-to-end adaptive preprocessing pipeline</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-2">
                {/* Flow steps */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {[
                    { step: '01', icon: Music, label: 'Raw Audio', desc: 'FMA track loaded — may contain background noise, hiss, or distortion', color: 'from-slate-500 to-slate-600' },
                    { step: '02', icon: ScanSearch, label: 'Audio Analysis', desc: 'SNR, spectral flatness, harmonic ratio and RMS energy are extracted', color: 'from-blue-500 to-cyan-500' },
                    { step: '03', icon: Brain, label: 'Routing Decision', desc: 'Trained MLP selects the best denoising method for this specific track', color: 'from-purple-500 to-pink-500' },
                    { step: '04', icon: Zap, label: 'Preprocessing', desc: 'Selected method applied — spectral gating, HPSS, Wiener filter, etc.', color: 'from-orange-500 to-yellow-500' },
                    { step: '05', icon: BarChart2, label: 'Classification', desc: 'Cleaned audio classified by CNN — accuracy measured vs. baseline', color: 'from-emerald-500 to-green-500' },
                  ].map(({ step, icon: Icon, label, desc, color }, i, arr) => (
                    <React.Fragment key={step}>
                      <div className="flex flex-col items-center text-center group">
                        <div className={`relative w-14 h-14 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg mb-3 group-hover:scale-110 transition-transform duration-300`}>
                          <Icon className="h-6 w-6 text-white" />
                          <span className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-background border-2 border-border flex items-center justify-center text-[9px] font-bold text-muted-foreground">{step}</span>
                        </div>
                        <p className="text-sm font-semibold mb-1">{label}</p>
                        <p className="text-xs text-muted-foreground leading-snug">{desc}</p>
                      </div>
                      {i < arr.length - 1 && (
                        <div className="hidden md:flex items-center justify-center col-span-0 self-center -mx-2 mt-[-40px]">
                          <ArrowRight className="h-4 w-4 text-muted-foreground/50" />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>

                {/* Routing method pills */}
                <div className="mt-8 pt-6 border-t border-border/50">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Routing selects from these denoising strategies:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { name: 'No Processing', desc: 'Passthrough for clean audio', color: 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300' },
                      { name: 'Spectral Gating', desc: 'Estimates stationary noise floor and suppresses it', color: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' },
                      { name: 'HPSS', desc: 'Separates harmonic and percussive content', color: 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' },
                      { name: 'Wiener Filter', desc: 'Gentle smoothing for broadband noise', color: 'bg-pink-100 dark:bg-pink-900/40 text-pink-700 dark:text-pink-300' },
                      { name: 'Spectral Subtraction', desc: 'Subtracts estimated noise spectrum', color: 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300' },
                    ].map(({ name, desc, color }) => (
                      <div key={name} className={`px-3 py-2 rounded-xl text-xs font-medium ${color} cursor-default hover:scale-105 transition-transform duration-200`} title={desc}>
                        {name}
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* ── Use Cases ────────────────────────────────────────────────── */}
            <div>
              <div className="text-center mb-6">
                <h3 className="text-xl font-bold mb-1">Where This System Can Be Applied</h3>
                <p className="text-sm text-muted-foreground">Adaptive preprocessing has real-world impact across many audio scenarios</p>
              </div>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  {
                    icon: Radio,
                    title: 'Broadcast & Streaming',
                    desc: 'Automatically improve audio quality of user-uploaded tracks from variable-quality sources before indexing.',
                    examples: ['Podcast platforms', 'Music streaming ingestion', 'Radio archive digitisation'],
                    gradient: 'from-blue-500 to-cyan-500',
                    bg: 'from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20',
                    border: 'border-blue-100 dark:border-blue-900/30',
                  },
                  {
                    icon: Mic2,
                    title: 'Field Recording Analysis',
                    desc: 'Classify music recorded at live events, outdoor venues, or studios where background noise is unavoidable.',
                    examples: ['Concert archiving', 'Ethnomusicology research', 'Live event tagging'],
                    gradient: 'from-purple-500 to-pink-500',
                    bg: 'from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20',
                    border: 'border-purple-100 dark:border-purple-900/30',
                  },
                  {
                    icon: Headphones,
                    title: 'Music Library Curation',
                    desc: 'Rapidly tag and categorise large digitised music collections where recording quality varies widely.',
                    examples: ['Digital archiving projects', 'Legacy library restoration', 'Music recommendation systems'],
                    gradient: 'from-emerald-500 to-teal-500',
                    bg: 'from-emerald-50 to-teal-50 dark:from-emerald-950/20 dark:to-teal-950/20',
                    border: 'border-emerald-100 dark:border-emerald-900/30',
                  },
                  {
                    icon: ShieldCheck,
                    title: 'Robust AI Pipelines',
                    desc: 'Make downstream classification models more resilient to real-world data degradation without retraining.',
                    examples: ['Audio ML benchmarking', 'Noisy-environment deployment', 'Model robustness research'],
                    gradient: 'from-orange-500 to-yellow-500',
                    bg: 'from-orange-50 to-yellow-50 dark:from-orange-950/20 dark:to-yellow-950/20',
                    border: 'border-orange-100 dark:border-orange-900/30',
                  },
                  {
                    icon: ScanSearch,
                    title: 'Audio Forensics',
                    desc: 'Identify genre or content markers from degraded recordings where original quality cannot be guaranteed.',
                    examples: ['Evidence audio analysis', 'Copyright detection', 'Watermark recovery'],
                    gradient: 'from-red-500 to-rose-500',
                    bg: 'from-red-50 to-rose-50 dark:from-red-950/20 dark:to-rose-950/20',
                    border: 'border-red-100 dark:border-red-900/30',
                  },
                  {
                    icon: Sparkles,
                    title: 'Research Platform',
                    desc: 'Benchmark custom preprocessing strategies against rule-based and no-preprocessing baselines with statistical rigour.',
                    examples: ['FYP / MSc experiments', 'ISMIR / ICASSP benchmarks', 'Ablation studies'],
                    gradient: 'from-violet-500 to-indigo-500',
                    bg: 'from-violet-50 to-indigo-50 dark:from-violet-950/20 dark:to-indigo-950/20',
                    border: 'border-violet-100 dark:border-violet-900/30',
                  },
                ].map(({ icon: Icon, title, desc, examples, gradient, bg, border }) => (
                  <div key={title} className={`p-5 rounded-2xl bg-gradient-to-br ${bg} border ${border} hover:shadow-lg transition-all duration-300 group hover:-translate-y-1`}>
                    <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${gradient} mb-4 shadow-md group-hover:scale-110 transition-transform duration-300`}>
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <h4 className="font-bold text-base mb-2">{title}</h4>
                    <p className="text-sm text-muted-foreground leading-relaxed mb-3">{desc}</p>
                    <div className="space-y-1">
                      {examples.map(ex => (
                        <div key={ex} className="flex items-center gap-2 text-xs text-muted-foreground">
                          <ChevronRight className="h-3 w-3 text-primary flex-shrink-0" />
                          {ex}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Status + CTA ─────────────────────────────────────────────── */}
            <div className="grid md:grid-cols-3 gap-4">
              <Card className="md:col-span-2 border-0 shadow-lg bg-gradient-to-br from-background to-muted/20">
                <CardContent className="flex items-center gap-4 py-5">
                  <div className={`p-3 rounded-xl ${isRunning ? 'bg-blue-500 animate-pulse' : experimentStatus.status === 'complete' ? 'bg-emerald-500' : 'bg-muted'}`}>
                    {isRunning ? <Loader2 className="h-5 w-5 text-white animate-spin" /> :
                     experimentStatus.status === 'complete' ? <CheckCircle2 className="h-5 w-5 text-white" /> :
                     <Gauge className="h-5 w-5 text-muted-foreground" />}
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold capitalize">{experimentStatus.status === 'idle' ? 'System Ready' : experimentStatus.status.replace(/_/g, ' ')}</p>
                    <p className="text-sm text-muted-foreground">{experimentStatus.message || 'Configure parameters and start the experiment to see live results'}</p>
                  </div>
                  {isRunning && (
                    <div className="flex items-center gap-1">
                      {[...Array(4)].map((_, i) => (
                        <div key={i} className="w-1 bg-blue-500 rounded-full animate-bounce" style={{ height: `${12 + i * 4}px`, animationDelay: `${i * 0.1}s` }} />
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
              <Button
                size="lg"
                className="h-full min-h-[64px] text-base font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-xl btn-glow"
                onClick={() => setActiveTab('experiment')}
                disabled={isRunning}
              >
                {isRunning ? (
                  <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Running...</>
                ) : (
                  <><Play className="mr-2 h-5 w-5" />Start Experiment</>
                )}
              </Button>
            </div>

          </TabsContent>

          {/* Experiment Tab */}
          <TabsContent value="experiment" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-3">
              {/* Configuration Panel */}
              <Card className="lg:col-span-1 border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500">
                      <Settings className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <CardTitle>Configuration</CardTitle>
                      <CardDescription>Set experiment parameters</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-5">
                  {[
                    { label: 'Max Samples', key: 'max_samples', desc: 'Audio samples to load' },
                    { label: 'Training Epochs', key: 'epochs', desc: 'CNN training iterations' },
                    { label: 'Routing Samples', key: 'routing_samples', desc: 'Samples for routing model' }
                  ].map((field) => (
                    <div key={field.key} className="space-y-2">
                      <label className="text-sm font-medium flex items-center gap-2">
                        <Zap className="h-3 w-3 text-primary" />
                        {field.label}
                      </label>
                      <input
                        type="number"
                        value={experimentParams[field.key]}
                        onChange={(e) => setExperimentParams(p => ({ ...p, [field.key]: parseInt(e.target.value) }))}
                        className="w-full px-4 py-3 rounded-xl border-2 bg-background/50 focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all duration-300 outline-none"
                        disabled={isRunning}
                      />
                      <p className="text-xs text-muted-foreground">{field.desc}</p>
                    </div>
                  ))}
                  {/* Test Noise SNR */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Waves className="h-3 w-3 text-primary" />
                      Test Noise Level
                    </label>
                    <select
                      value={experimentParams.test_noise_snr}
                      onChange={(e) => setExperimentParams(p => ({ ...p, test_noise_snr: Number(e.target.value) }))}
                      className="w-full px-4 py-3 rounded-xl border-2 bg-background/50 focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all duration-300 outline-none"
                      disabled={isRunning}
                    >
                      <option value={0}>Clean (no noise)</option>
                      <option value={20}>20 dB SNR (light)</option>
                      <option value={15}>15 dB SNR (moderate)</option>
                      <option value={10}>10 dB SNR (heavy)</option>
                      <option value={5}>5 dB SNR (severe)</option>
                    </select>
                    <p className="text-xs text-muted-foreground">Noise added to test audio only — shows adaptive system value</p>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button 
                    className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-xl btn-glow"
                    onClick={startExperiment}
                    disabled={isRunning || !connected}
                  >
                    {isRunning ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        Running Experiment...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-5 w-5" />
                        Start Experiment
                      </>
                    )}
                  </Button>
                </CardFooter>
              </Card>

              {/* Progress Panel */}
              <Card className="lg:col-span-2 border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in animation-delay-100">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${isRunning ? 'bg-gradient-to-br from-blue-500 to-purple-500 animate-pulse' : 'bg-muted'}`}>
                        {isRunning ? (
                          <Loader2 className="h-5 w-5 text-white animate-spin" />
                        ) : experimentStatus.status === 'complete' ? (
                          <CheckCircle2 className="h-5 w-5 text-green-500" />
                        ) : (
                          <Timer className="h-5 w-5 text-muted-foreground" />
                        )}
                      </div>
                      <div>
                        <CardTitle>Experiment Progress</CardTitle>
                        <CardDescription>
                          {experimentStatus.status === 'idle' ? 'Ready to start' : experimentStatus.message}
                        </CardDescription>
                      </div>
                    </div>
                    {isRunning && (
                      <div className="flex items-center gap-2">
                        <WaveformAnimation active={isRunning} />
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-8">
                  {/* Progress Bar */}
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium capitalize">{experimentStatus.status.replace(/_/g, ' ')}</span>
                      <span className="font-bold text-primary">
                        <AnimatedNumber value={experimentStatus.progress} suffix="%" />
                      </span>
                    </div>
                    <div className="relative h-4 rounded-full bg-muted overflow-hidden">
                      <div 
                        className={`h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 transition-all duration-500 ${isRunning ? 'progress-striped progress-animated' : ''}`}
                        style={{ width: `${experimentStatus.progress}%` }}
                      />
                    </div>
                  </div>

                  {/* Progress Steps */}
                  <ProgressSteps 
                    currentStep={Math.max(0, getCurrentStepIndex() - 1)}
                    steps={progressSteps}
                  />

                  {/* Logs */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <Activity className="h-4 w-4 text-primary" />
                      Live Logs
                    </h4>
                    <div className="h-52 overflow-y-auto rounded-xl border bg-black/5 dark:bg-white/5 p-4 font-mono text-xs space-y-1">
                      {logs.length === 0 ? (
                        <p className="text-muted-foreground flex items-center gap-2">
                          <Loader2 className="h-3 w-3" />
                          Waiting for experiment to start...
                        </p>
                      ) : (
                        logs.map((log, i) => (
                          <div key={i} className="animate-fade-in flex gap-3 py-1 border-b border-border/30 last:border-0">
                            <span className="text-muted-foreground flex-shrink-0">{log.time}</span>
                            <span className="text-foreground">{log.message}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results" className="space-y-6">
            {results ? (
              <>
                {/* Results Summary Cards */}
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 stagger-children">
                  <div className="animate-fade-in-up opacity-0" style={{ animationDelay: '0ms', animationFillMode: 'forwards' }}>
                    <Card className="relative overflow-hidden border-0 shadow-xl bg-gradient-to-br from-blue-500 to-blue-600 text-white">
                      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-30" />
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-white/80">Baseline Accuracy</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-4xl font-bold">
                          <AnimatedNumber value={results.baseline.accuracy * 100} decimals={1} suffix="%" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="animate-fade-in-up opacity-0" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                    <Card className="relative overflow-hidden border-0 shadow-xl bg-gradient-to-br from-green-500 to-emerald-600 text-white">
                      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-30" />
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-white/80">Adaptive Accuracy</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-4xl font-bold">
                          <AnimatedNumber value={results.adaptive.accuracy * 100} decimals={1} suffix="%" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="animate-fade-in-up opacity-0" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                    <Card className="relative overflow-hidden border-0 shadow-xl bg-gradient-to-br from-purple-500 to-pink-600 text-white">
                      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-30" />
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-white/80">Improvement</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-4xl font-bold flex items-center gap-2">
                          <TrendingUp className="h-8 w-8" />
                          <AnimatedNumber value={results.improvements.accuracy} decimals={2} suffix="%" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="animate-fade-in-up opacity-0" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
                    <Card className="relative overflow-hidden border-0 shadow-xl bg-gradient-to-br from-orange-500 to-yellow-500 text-white">
                      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-30" />
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-white/80">F1 Score Gain</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-4xl font-bold">
                          +<AnimatedNumber value={results.improvements.f1_macro} decimals={2} suffix="%" />
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>

                {/* Routing Accuracy + Test Condition Banner */}
                <div className="grid gap-4 md:grid-cols-2 animate-fade-in">
                  <Card className="border-0 shadow-lg bg-gradient-to-br from-cyan-500 to-blue-600 text-white">
                    <CardContent className="flex items-center gap-4 py-4">
                      <div className="p-3 rounded-xl bg-white/20">
                        <Brain className="h-6 w-6" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white/80">Routing Model Accuracy</p>
                        <p className="text-3xl font-bold">
                          {results.routing_accuracy != null
                            ? (results.routing_accuracy * 100).toFixed(1) + '%'
                            : 'N/A'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                  <Card className="border-0 shadow-lg bg-gradient-to-br from-violet-500 to-purple-700 text-white">
                    <CardContent className="flex items-center gap-4 py-4">
                      <div className="p-3 rounded-xl bg-white/20">
                        <Waves className="h-6 w-6" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white/80">Test Noise Condition</p>
                        <p className="text-3xl font-bold">
                          {results.config?.test_noise_snr > 0
                            ? `${results.config.test_noise_snr} dB SNR`
                            : 'Clean'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Charts Section */}
                <div className="grid gap-6 lg:grid-cols-2">
                  {/* Metrics Comparison Chart */}
                  <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500">
                          <BarChart3 className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <CardTitle>Metrics Comparison</CardTitle>
                          <CardDescription>Baseline vs Adaptive performance</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={[
                          { metric: 'Accuracy', Baseline: results.baseline.accuracy, Adaptive: results.adaptive.accuracy },
                          { metric: 'F1 Score', Baseline: results.baseline.f1_macro, Adaptive: results.adaptive.f1_macro },
                          { metric: 'AUC-ROC', Baseline: results.baseline.auc_roc, Adaptive: results.adaptive.auc_roc },
                          { metric: 'Precision', Baseline: results.baseline.precision, Adaptive: results.adaptive.precision },
                          { metric: 'Recall', Baseline: results.baseline.recall, Adaptive: results.adaptive.recall },
                        ]} barGap={8}>
                          <defs>
                            <linearGradient id="baselineGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#3b82f6" />
                              <stop offset="100%" stopColor="#1d4ed8" />
                            </linearGradient>
                            <linearGradient id="adaptiveGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#10b981" />
                              <stop offset="100%" stopColor="#059669" />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted/50" />
                          <XAxis dataKey="metric" className="text-xs" tick={{ fill: 'currentColor' }} />
                          <YAxis domain={[0, 1]} className="text-xs" tick={{ fill: 'currentColor' }} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '12px',
                              boxShadow: '0 10px 40px rgba(0,0,0,0.1)'
                            }}
                            formatter={(value) => (value * 100).toFixed(1) + '%'}
                          />
                          <Legend />
                          <Bar dataKey="Baseline" fill="url(#baselineGradient)" radius={[6, 6, 0, 0]} />
                          <Bar dataKey="Adaptive" fill="url(#adaptiveGradient)" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Preprocessing Distribution */}
                  <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in animation-delay-100">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
                          <Zap className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <CardTitle>Preprocessing Distribution</CardTitle>
                          <CardDescription>How the routing model assigned methods</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={320}>
                        <PieChart>
                          <defs>
                            {COLORS.map((color, i) => (
                              <linearGradient key={i} id={`pieGradient${i}`} x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stopColor={color} />
                                <stop offset="100%" stopColor={color} stopOpacity={0.7} />
                              </linearGradient>
                            ))}
                          </defs>
                          <Pie
                            data={Object.entries(results.preprocessing_distribution)
                              .filter(([_, count]) => count > 0)
                              .map(([name, count]) => ({ name: name.replace(/_/g, ' '), value: count }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                            outerRadius={100}
                            innerRadius={40}
                            paddingAngle={2}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {Object.entries(results.preprocessing_distribution)
                              .filter(([_, count]) => count > 0)
                              .map((_, index) => (
                                <Cell key={`cell-${index}`} fill={`url(#pieGradient${index % COLORS.length})`} />
                              ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '12px',
                              boxShadow: '0 10px 40px rgba(0,0,0,0.1)'
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Routing Confidence Distribution */}
                  {results.routing_decisions && results.routing_decisions.length > 0 && (() => {
                    const confBuckets = [0, 0, 0, 0, 0]
                    results.routing_decisions.forEach(d => {
                      const idx = Math.min(4, Math.floor(d.confidence * 5))
                      confBuckets[idx]++
                    })
                    const confData = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'].map((label, i) => ({
                      range: label, count: confBuckets[i]
                    }))
                    return (
                      <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in animation-delay-100">
                        <CardHeader>
                          <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500">
                              <Brain className="h-5 w-5 text-white" />
                            </div>
                            <div>
                              <CardTitle>Routing Confidence Distribution</CardTitle>
                              <CardDescription>How confident the routing model was per sample</CardDescription>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <ResponsiveContainer width="100%" height={240}>
                            <BarChart data={confData}>
                              <CartesianGrid strokeDasharray="3 3" className="stroke-muted/50" />
                              <XAxis dataKey="range" className="text-xs" tick={{ fill: 'currentColor' }} />
                              <YAxis className="text-xs" tick={{ fill: 'currentColor' }} />
                              <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '12px' }} />
                              <Bar dataKey="count" name="Samples" fill="#06b6d4" radius={[6, 6, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    )
                  })()}

                  {/* Per-Genre Performance */}
                  <Card className="lg:col-span-2 border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in animation-delay-200">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-orange-500 to-yellow-500">
                          <Music className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <CardTitle>Per-Genre F1 Score Comparison</CardTitle>
                          <CardDescription>Performance breakdown by music genre</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={results.per_genre} layout="vertical" barGap={4}>
                          <defs>
                            <linearGradient id="genreBaselineGradient" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%" stopColor="#3b82f6" />
                              <stop offset="100%" stopColor="#60a5fa" />
                            </linearGradient>
                            <linearGradient id="genreAdaptiveGradient" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%" stopColor="#10b981" />
                              <stop offset="100%" stopColor="#34d399" />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted/50" />
                          <XAxis type="number" domain={[0, 1]} className="text-xs" tick={{ fill: 'currentColor' }} />
                          <YAxis dataKey="genre" type="category" width={100} className="text-xs" tick={{ fill: 'currentColor' }} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '12px',
                              boxShadow: '0 10px 40px rgba(0,0,0,0.1)'
                            }}
                            formatter={(value) => (value * 100).toFixed(1) + '%'}
                          />
                          <Legend />
                          <Bar dataKey="baseline_f1" name="Baseline" fill="url(#genreBaselineGradient)" radius={[0, 6, 6, 0]} />
                          <Bar dataKey="adaptive_f1" name="Adaptive" fill="url(#genreAdaptiveGradient)" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>

                {/* Export Section */}
                <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30 animate-fade-in">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500">
                        <Download className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <CardTitle>Export Results</CardTitle>
                        <CardDescription>Download experiment data in various formats</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="flex gap-4">
                    <Button onClick={() => api.exportJSON()} className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 shadow-lg btn-shine">
                      <Download className="mr-2 h-4 w-4" />
                      Export JSON
                    </Button>
                    <Button onClick={() => api.exportCSV()} variant="outline" className="hover-lift">
                      <Download className="mr-2 h-4 w-4" />
                      Export CSV
                    </Button>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="border-0 shadow-xl bg-gradient-to-br from-background to-muted/30">
                <CardContent className="flex flex-col items-center justify-center py-20">
                  <div className="p-6 rounded-full bg-muted/50 mb-6 animate-bounce-subtle">
                    <BarChart3 className="h-16 w-16 text-muted-foreground" />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">No Results Available</h3>
                  <p className="text-muted-foreground text-center max-w-md mb-6">
                    Run an experiment to see beautiful visualizations of your model's performance here
                  </p>
                  <Button 
                    onClick={() => setActiveTab('experiment')}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-xl btn-shine"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Start Experiment
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 bg-muted/30">
        <div className="container mx-auto px-4 py-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg">
              <Waves className="h-4 w-4 text-white" />
            </div>
            <span className="font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Adaptive Audio Preprocessing
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            v2.0 • Model-Based Routing • Neural Network Preprocessing Selection • {new Date().getFullYear()}
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
