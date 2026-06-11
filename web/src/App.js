import { useState, useCallback } from "react";
import ConfidenceBars from "./components/ConfidenceBars";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.5);
  const [maxLength, setMaxLength] = useState(12);
  const [generated, setGenerated] = useState("");
  const [topWords, setTopWords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [slowStart, setSlowStart] = useState(false);
  const [error, setError] = useState("");

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setSlowStart(false);
    setError("");
    const warmupTimer = setTimeout(() => setSlowStart(true), 8000);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim(), temperature, max_length: maxLength }),
        signal: controller.signal,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setGenerated(data.generated);
      setTopWords(data.top_words);
    } catch (e) {
      if (e.name === "AbortError") {
        setError("Request timed out. The model endpoint may be unavailable.");
      } else {
        setError(e.message);
      }
    } finally {
      clearTimeout(warmupTimer);
      clearTimeout(timeoutId);
      setLoading(false);
      setSlowStart(false);
    }
  }, [prompt, temperature, maxLength]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const handleClear = () => {
    setPrompt("");
    setGenerated("");
    setTopWords([]);
    setError("");
  };

  return (
    <div className="app">
      <h1 className="title">GPT-Nano Text Generation</h1>
      <div className="subtitle">Causal transformer trained on WikiText-2 · 7M params · BPE tokenization</div>
      <div className="lesson-note">
        Built from scratch in PyTorch — causal self-attention, weight tying, cosine LR decay.
        Trained on GPU via a full AWS ML pipeline: S3 → SageMaker → Serverless Endpoint → Lambda + API Gateway.
      </div>
      <div className="top-btn-row">
        <a
          className="btn-outline"
          href="https://github.com/sassom2112/lstm-text-prediction"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </div>

      <div className="main-grid">
        {/* Left: input panel */}
        <div className="panel">
          <div className="panel-label">PROMPT</div>
          <textarea
            className="prompt-input"
            placeholder={"Type a prompt and press Enter…\n\nTry: the history of, in the early, scientists discovered"}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
          />

          <div className="controls">
            <div className="control-row">
              <label>
                Temperature <span className="val">{temperature.toFixed(2)}</span>
              </label>
              <input
                type="range" min="0" max="1.5" step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="slider"
              />
              <div className="range-labels">
                <span>Greedy (0)</span>
                <span>Creative (1.5)</span>
              </div>
            </div>
            <div className="control-row">
              <label>
                Max new words <span className="val">{maxLength}</span>
              </label>
              <input
                type="range" min="1" max="30" step="1"
                value={maxLength}
                onChange={(e) => setMaxLength(parseInt(e.target.value))}
                className="slider"
              />
            </div>
          </div>

          <div className="action-row">
            <button
              className="btn-primary"
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
            >
              {loading ? (slowStart ? "Warming up model…" : "Generating…") : "Generate"}
            </button>
            <button className="btn-secondary" onClick={handleClear}>
              Clear
            </button>
          </div>

          {loading && slowStart && (
            <div className="cold-start-note">
              SageMaker Serverless endpoint is cold — first request takes ~15–30s.
            </div>
          )}

          {error && <div className="error-msg">{error}</div>}

          {generated && (
            <div className="output-box">
              <div className="panel-label">GENERATED</div>
              <div className="output-text">{generated}</div>
            </div>
          )}
        </div>

        {/* Right: next-word probability bars */}
        <div className="panel">
          <div className="panel-label">NEXT TOKEN PROBABILITIES</div>
          <ConfidenceBars words={topWords} />
        </div>
      </div>
    </div>
  );
}
