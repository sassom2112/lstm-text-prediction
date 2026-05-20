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
  const [error, setError] = useState("");

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim(), temperature, max_length: maxLength }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setGenerated(data.generated);
      setTopWords(data.top_words);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
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
      <h1 className="title">LSTM Text Generation</h1>
      <div className="subtitle">Word-level sequence model trained on English sentences</div>
      <div className="lesson-note">
        Trained on 3,000 short sentences — intentionally minimal to show what a baseline LSTM learns,
        and why attention mechanisms and transformers exist.
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
            placeholder={"Type a prompt and press Enter…\n\nTry: i am, she wants to, the cat"}
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
              {loading ? "Generating…" : "Generate"}
            </button>
            <button className="btn-secondary" onClick={handleClear}>
              Clear
            </button>
          </div>

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
          <div className="panel-label">NEXT WORD PROBABILITIES</div>
          <ConfidenceBars words={topWords} />
        </div>
      </div>
    </div>
  );
}
