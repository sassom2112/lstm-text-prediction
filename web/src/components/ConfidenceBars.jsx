export default function ConfidenceBars({ words }) {
  if (!words || words.length === 0) {
    return (
      <div className="bars-empty">
        Generate text to see next-word probabilities
      </div>
    );
  }

  const max = Math.max(...words.map((w) => w.prob));

  return (
    <div className="bars">
      {words.map(({ word, prob }) => {
        const pct = max > 0 ? (prob / max) * 100 : 0;
        const isTop = prob === max;
        return (
          <div key={word} className={`bar-row ${isTop ? "bar-row--top" : ""}`}>
            <span className="bar-word">{word}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="bar-pct">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
