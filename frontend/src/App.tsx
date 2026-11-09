
import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

type TelemetryPoint = {
  timestep: number
  value: number
  anomaly_score: number
  is_anomaly: boolean
}

type InferenceResponse = {
  model: string
  threshold: number
  total_predictions: number
  points: TelemetryPoint[]
}

export default function App() {
  const [data, setData] = useState<InferenceResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/inference/A-2?use_commands=true&start=4300&limit=300')
      .then((response) => {
        if (!response.ok) throw new Error('Inference request failed')
        return response.json()
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [])

  if (error) return <p>Error: {error}</p>
  if (!data) return <p>Loading NASA telemetry...</p>

  const alerts = data.points.filter((point) => point.is_anomaly).length

  return (
    <main style={{ padding: 32, maxWidth: 1200, margin: 'auto' }}>
      <h1>MissionGuard</h1>
      <p>NASA Spacecraft Anomaly Detection</p>

      <section>
        <h2>SMAP — Channel A-2</h2>
        <p>Model: {data.model}</p>
        <p>
          {data.points.length} telemetry readings | {alerts} flagged anomalies
        </p>
      </section>

      <h3>Spacecraft Telemetry</h3>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={data.points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestep" />
            <YAxis domain={[-1, 1]} />
            <Tooltip />
            <Line
              type="linear"
              dataKey="value"
              stroke="#60a5fa"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h3>LSTM Anomaly Scores</h3>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={data.points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestep" />
            <YAxis />
            <Tooltip />
            <ReferenceLine
              y={data.threshold}
              stroke="#ef4444"
              strokeDasharray="5 5"
            />
            <Line
              type="linear"
              dataKey="anomaly_score"
              stroke="#a78bfa"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </main>
  )
}
