export default function Simulation({ patientData, simulationResult, onSelectScenario }) {
  if (!simulationResult) {
    return <div className="container">시뮬레이션 결과를 불러오는 중...</div>;
  }
  if (!simulationResult.drugs || simulationResult.drugs.length === 0) {
    return <div className="container">결과가 없습니다. 입력값을 확인해주세요.</div>;
  }

  const grid = { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 };

  // 약물 하나를 시나리오 객체로 변환 (Summary와 호환)
  const asScenario = (drugItem) => {
    const effect = Math.round((drugItem.weeks?.at(-1)?.score || 0) * 100);
    return {
      effect,
      _selectedDrug: drugItem.drug,
    };
  };

  return (
    <div className="container">
      <h1>시뮬레이션 결과</h1>

      <h3>약물별 최종 확률(%)</h3>
      <ul>
        {simulationResult.probabilities?.map((p) => (
          <li key={p.drug}>
            {p.drug}: {p.percent.toFixed(1)}% {p.recommended ? "(추천)" : ""}
          </li>
        ))}
      </ul>

      <h3>주차별 이미지 (12주)</h3>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {simulationResult.drugs.map((drug) => (
          <div
            key={drug.drug}
            style={{ border: "1px solid #ccc", padding: 16, borderRadius: 8, minWidth: 260 }}
          >
            <h4>
              {drug.drug} {drug.is_best ? "(추천)" : ""}
            </h4>

            <div style={grid}>
              {drug.weeks.map((w) => (
                <div key={w.week} style={{ border: "1px solid #eee", padding: 8 }}>
                  <div style={{ fontSize: 12, marginBottom: 6 }}>
                    {w.week}주차
                  </div>
                  <img
                    src={w.image_url}
                    alt={`${drug.drug}-${w.week}`}
                    style={{ width: "100%", height: "auto", display: "block" }}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 다음 버튼 */}
      <div style={{ marginTop: 24, textAlign: "center" }}>
        <button
          onClick={() => {
            const bestDrug = simulationResult.drugs.find((d) => d.is_best) || simulationResult.drugs[0];
            onSelectScenario(asScenario(bestDrug));
          }}
        >
          결과 요약
        </button>
      </div>
    </div>
  );
}



