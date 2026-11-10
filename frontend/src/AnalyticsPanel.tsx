import { useEffect, useState } from "react";
import { apiUrl } from "./api";

type EvaluationRow = {
  Model: string;
  Precision: number;
  Recall: number;
  F1: number;
  TP: number;
  FP: number;
  FN: number;
  TN: number;
};

type EventRow = {
  Model: string;
  "Labeled Events": number;
  "Detected Events": number;
  "Event Recall": number;
  "False Alert Runs": number;
  "Median Detection Delay": number | null;
};

type Props = {
  models: string[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
};

export default function AnalyticsPanel({
  models,
  selectedModel,
  onSelectModel,
}: Props) {
  const [metrics, setMetrics] = useState<EvaluationRow[]>([]);
  const [events, setEvents] = useState<EventRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const match = selectedModel.match(/^(.+)_h(\d+)(_cmd)?$/);
    if (!match) return;

    const controller = new AbortController();
    const channel = match[1];

    const params = new URLSearchParams({
      horizon: match[2],
      use_commands: String(Boolean(match[3])),
    });

    async function load() {
      setLoading(true);
      setError("");
      setMetrics([]);
      setEvents(null);

      try {
        const base = apiUrl(`/evaluation/${encodeURIComponent(channel)}`);

        const [metricResponse, eventResponse] = await Promise.all([
          fetch(`${base}?${params}`, { signal: controller.signal }),
          fetch(`${base}/events?${params}`, { signal: controller.signal }),
        ]);

        if (!metricResponse.ok) {
          throw new Error("Model evaluation report not found");
        }

        const metricData: { results: EvaluationRow[] } =
          await metricResponse.json();

        let eventRows: EventRow[] | null = null;

        if (eventResponse.ok) {
          const eventData: { results: EventRow[] } = await eventResponse.json();
          eventRows = eventData.results;
        } else if (eventResponse.status !== 404) {
          throw new Error("Failed to load event evaluation");
        }

        if (!controller.signal.aborted) {
          setMetrics(metricData.results);
          setEvents(eventRows);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Request failed");
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    load();
    return () => controller.abort();
  }, [selectedModel]);

  const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="analytics-page">
      <section className="control-panel">
        <div className="control-group">
          <label htmlFor="analytics-model">Compare model</label>
          <select
            id="analytics-model"
            value={selectedModel}
            onChange={(e) => onSelectModel(e.target.value)}
          >
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
        <div className="control-note">
          Evaluation against NASA's labeled anomaly intervals
        </div>
      </section>

      {loading && (
        <div className="empty-state">Loading model evaluation...</div>
      )}
      {error && <div className="error">{error}</div>}

      {!loading && !error && metrics.length > 0 && (
        <>
          <section className="analytics-card">
            <div className="panel-heading">
              <div>
                <h2>Model Performance</h2>
                <p>Statistical baseline versus trained LSTM</p>
              </div>
            </div>

            <div className="analytics-metrics">
              {metrics.map((row) => (
                <div className="analytics-metric" key={row.Model}>
                  <div className="kpi-label">{row.Model}</div>
                  <div className="kpi-value">{percent(row.F1)}</div>
                  <div className="kpi-foot">Timestep-level F1 score</div>
                </div>
              ))}
            </div>

            <div className="analytics-bars">
              {metrics.map((row) => (
                <div key={row.Model}>
                  <div className="analytics-bar-label">
                    <span>{row.Model}</span>
                    <strong>{percent(row.F1)}</strong>
                  </div>
                  <div className="analytics-bar-track">
                    <div
                      className="analytics-bar-fill"
                      style={{ width: `${row.F1 * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="analytics-card">
            <div className="panel-heading">
              <div>
                <h2>Classification Metrics</h2>
                <p>Performance across evaluated telemetry timesteps</p>
              </div>
            </div>

            <div className="analytics-table-wrap">
              <table className="analytics-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((row) => (
                    <tr key={row.Model}>
                      <td>{row.Model}</td>
                      <td>{percent(row.Precision)}</td>
                      <td>{percent(row.Recall)}</td>
                      <td>{percent(row.F1)}</td>
                      <td>{row.TP}</td>
                      <td>{row.FP}</td>
                      <td>{row.FN}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="analytics-card">
            <div className="panel-heading">
              <div>
                <h2>Incident-Level Detection</h2>
                <p>
                  Detected anomaly events, false alert runs, and detection delay
                </p>
              </div>
            </div>

            {events ? (
              <div className="analytics-table-wrap">
                <table className="analytics-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Detected</th>
                      <th>Event Recall</th>
                      <th>False Alerts</th>
                      <th>Median Delay</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((row) => (
                      <tr key={row.Model}>
                        <td>{row.Model}</td>
                        <td>
                          {row["Detected Events"]}/{row["Labeled Events"]}
                        </td>
                        <td>{percent(row["Event Recall"])}</td>
                        <td>{row["False Alert Runs"]}</td>
                        <td>
                          {row["Median Detection Delay"] == null
                            ? "N/A"
                            : `${row["Median Detection Delay"]} steps`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="subtext">
                No event-level report is available for this checkpoint yet.
              </p>
            )}
          </section>

          <p className="analytics-disclaimer">
            Research benchmark results on previously explored channels. Event
            recall counts an incident as detected if at least one timestep is
            flagged. These results are not an unbiased estimate of performance
            on new spacecraft data.
          </p>
        </>
      )}
    </div>
  );
}
