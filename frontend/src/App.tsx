
import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

type TelemetryPoint = {
  timestep: number
  value: number
  anomaly_score: number
  is_anomaly: boolean
}

type InferenceResponse = {
  channel: string
  model: string
  threshold: number
  total_predictions: number
  points: TelemetryPoint[]
}

const PAGE_SIZE = 300

export default function App() {
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('A-2_h1_cmd')
  const [start, setStart] = useState(4300)
  const [data, setData] = useState<InferenceResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const channel = selectedModel.split('_h')[0]
  const channels = [...new Set(models.map(m => m.split('_h')[0]))].sort()
  const availableModels = models.filter(m => m.startsWith(`${channel}_h`))

  // Load available checkpoints
  useEffect(() => {
    const controller = new AbortController()

    fetch('/api/models', { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error('Cannot load available models')
        return res.json()
      })
      .then(result => {
        const names: string[] = result.models
        setModels(names)
        setSelectedModel(current =>
          names.includes(current) ? current : (names[0] ?? '')
        )
      })
      .catch(err => {
        if (!controller.signal.aborted) setError(err.message)
      })

    return () => controller.abort()
  }, [])

  // Fetch model predictions when selection changes
  useEffect(() => {
    if (!selectedModel) return

    const match = selectedModel.match(/^(.+)_h(\d+)(_cmd)?$/)
    if (!match) return

    const controller = new AbortController()

    const params = new URLSearchParams({
      horizon: match[2],
      use_commands: String(Boolean(match[3])),
      start: String(start),
      limit: String(PAGE_SIZE),
    })

    setLoading(true)
    setError('')
    setData(null)

    fetch(`/api/inference/${channel}?${params}`, {
      signal: controller.signal,
    })
      .then(res => {
        if (!res.ok) throw new Error(`Inference failed (${res.status})`)
        return res.json()
      })
      .then((result: InferenceResponse) => setData(result))
      .catch(err => {
        if (!controller.signal.aborted) setError(err.message)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [selectedModel, start, channel])

  function changeChannel(next: string) {
    const choices = models.filter(m => m.startsWith(`${next}_h`))
    const preferred = choices.find(m => m === `${next}_h1_cmd`)
    setSelectedModel(preferred ?? choices[0] ?? '')
    setStart(next === 'A-2' ? 4300 : 0)
  }

  const points = data?.points ?? []
  const alerts = points.filter(p => p.is_anomaly)
  const lastPage = data ? start + PAGE_SIZE >= data.total_predictions : true

  const tooltipStyle = {
    backgroundColor: '#151e2c',
    border: '1px solid #293749',
    borderRadius: 8,
    color: '#e7edf6',
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <div className="brand-name">MissionGuard</div>
            <div className="brand-subtitle">SPACECRAFT INTELLIGENCE</div>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        <div className="nav-item nav-active">◉ &nbsp; Telemetry Monitor</div>
        <div className="nav-item">◇ &nbsp; Anomaly Detection</div>

        <div className="side-label">DATA SOURCE</div>
        <div className="side-detail">NASA JPL Telemanom</div>
        <div className="side-detail muted">SMAP / Curiosity</div>

        <div className="sidebar-bottom">
          <div className="side-detail"><span className="dot" /> Local API</div>
          <div className="side-detail muted">Historical telemetry replay</div>
          <div className="side-detail muted">MissionGuard v0.2</div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">MISSION OPERATIONS / MONITORING</div>
            <h1>Telemetry Overview</h1>
            <p className="subtext">
              Model predictions and anomaly analysis using NASA mission data
            </p>
          </div>
          <div className="status-badge">
            <span className="dot" />
            {data ? 'API CONNECTED' : 'LOCAL MODE'}
          </div>
        </header>

        <section className="control-panel">
          <div className="control-group">
            <label htmlFor="channel">Spacecraft channel</label>
            <select
              id="channel"
              value={channel}
              onChange={e => changeChannel(e.target.value)}
            >
              {channels.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="model">Detection model</label>
            <select
              id="model"
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
            >
              {availableModels.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          <div className="control-note">
            <span className="dot amber" />
            Historical replay, not live spacecraft data
          </div>
        </section>

        <section className="kpi-grid">
          <div className="kpi">
            <div className="kpi-label">VISIBLE READINGS</div>
            <div className="kpi-value">{points.length}</div>
            <div className="kpi-foot">Current telemetry window</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">FLAGGED TIMESTEPS</div>
            <div className="kpi-value alert-color">{alerts.length}</div>
            <div className="kpi-foot">Model-generated alerts</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">DETECTION THRESHOLD</div>
            <div className="kpi-value">
              {data ? data.threshold.toFixed(4) : '—'}
            </div>
            <div className="kpi-foot">Calibrated on normal validation data</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">TOTAL PREDICTIONS</div>
            <div className="kpi-value">
              {data ? data.total_predictions.toLocaleString() : '—'}
            </div>
            <div className="kpi-foot">Available historical scores</div>
          </div>
        </section>

        {error && <div className="error">{error}. Check your FastAPI server.</div>}
        {loading && <div className="empty-state">Loading spacecraft telemetry and predictions...</div>}

        {data && (
          <>
            <section className="chart-panel">
              <div className="panel-heading">
                <div>
                  <h2>Spacecraft Telemetry</h2>
                  <p>Normalized sensor measurements over time</p>
                </div>
                <span className="chart-legend">● Telemetry &nbsp; <span className="alert-color">● Alerts</span></span>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={points} margin={{ top: 12, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid stroke="#263143" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="timestep" type="number" domain={['dataMin', 'dataMax']} stroke="#8291a6" tickLine={false} axisLine={false} />
                    <YAxis domain={[-1.1, 1.1]} stroke="#8291a6" tickLine={false} axisLine={false} width={45} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="linear" dataKey="value" stroke="#5ca8f9" strokeWidth={2} dot={false} isAnimationActive={false} />
                    {alerts.map(point => (
                      <ReferenceDot
                        key={point.timestep}
                        x={point.timestep}
                        y={point.value}
                        r={3}
                        fill="#ff7388"
                        stroke="none"
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="chart-panel">
              <div className="panel-heading">
                <div>
                  <h2>Anomaly Detection Scores</h2>
                  <p>Forecast error compared against the calibrated threshold</p>
                </div>
                <span className="chart-legend">● Prediction error</span>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={points} margin={{ top: 12, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid stroke="#263143" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="timestep" type="number" domain={['dataMin', 'dataMax']} stroke="#8291a6" tickLine={false} axisLine={false} />
                    <YAxis stroke="#8291a6" tickLine={false} axisLine={false} width={45} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <ReferenceLine y={data.threshold} stroke="#ffb454" strokeDasharray="5 5" />
                    <Line type="linear" dataKey="anomaly_score" stroke="#a795f9" strokeWidth={2} dot={false} isAnimationActive={false} />
                    {alerts.map(point => (
                      <ReferenceDot
                        key={point.timestep}
                        x={point.timestep}
                        y={point.anomaly_score}
                        r={3}
                        fill="#ff7388"
                        stroke="none"
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="alerts-panel">
              <div className="panel-heading">
                <div>
                  <h2>Detected Anomalies</h2>
                  <p>Flagged telemetry readings within the current window</p>
                </div>
                <span className="alert-count">{alerts.length} alerts</span>
              </div>

              {alerts.length === 0 ? (
                <p className="subtext">No anomalies flagged in this window.</p>
              ) : (
                <div className="alerts-list">
                  {alerts.slice(0, 12).map(point => (
                    <div className="alert-row" key={point.timestep}>
                      <span className="alert-index">● &nbsp; Timestep {point.timestep}</span>
                      <span className="alert-score">Score: {point.anomaly_score.toFixed(4)}</span>
                    </div>
                  ))}
                  {alerts.length > 12 && (
                    <p className="subtext">Showing the first 12 of {alerts.length} flagged readings.</p>
                  )}
                </div>
              )}
            </section>

            <div className="footer-row">
              <div className="subtext">
                Window {start.toLocaleString()}–{(start + points.length - 1).toLocaleString()} of {data.total_predictions.toLocaleString()} scored windows
              </div>
              <div className="pager">
                <button disabled={start === 0} onClick={() => setStart(s => Math.max(0, s - PAGE_SIZE))}>
                  ← Previous
                </button>
                <button disabled={lastPage} onClick={() => setStart(s => s + PAGE_SIZE)}>
                  Next →
                </button>
              </div>
            </div>
          </>
        )}

        <footer className="footer-note">
          MissionGuard • Research prototype • Anomaly flags are model predictions, not verified spacecraft faults.
        </footer>
      </main>
    </div>
  )
}
