import { useEffect, useState } from "react";

export default function Summary({ patientData, scenario, reportId, onRestart, onBackToSimulation }) {
  const [report, setReport] = useState(null);

  // reportId로 백엔드에서 보고서 조회
  useEffect(() => {
    if (!reportId) return;
    fetch(`http://127.0.0.1:8000/api/report/${reportId}`)
      .then((res) => res.json())
      .then((data) => setReport(data))
      .catch((err) => console.error("보고서 불러오기 실패:", err));
  }, [reportId]);

  if (!scenario) {
    return <div className="container">❌ 시나리오가 선택되지 않았습니다.</div>;
  }

  const handleSaveServer = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/save-result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report_id: reportId,
          patientData,
          scenario,
          report,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        alert(`서버에 저장 완료!\n파일 경로: ${data.path}`);
      }
    } catch (err) {
      console.error("서버 저장 실패:", err);
    }
  };

  const handlePrint = () => window.print();

  return (
    <div className="container">
      <h1>결과 요약</h1>
      <p>선택한 방안에 대한 요약 결과입니다.</p>

      {/* 환자 정보 카드 */}
      {report && report.patient && (
        <div style={{ marginTop: 20 }}>
          <h2>환자 기본 정보</h2>
          <div
            style={{
              border: "1px solid #fff",
              borderRadius: 10,
              padding: 16,
              maxWidth: 400,
              backgroundColor: "#fff",
              boxShadow: "0 2px 6px rgba(0, 0, 0, 0.1)",
            }}
          >
            <p><strong>이름:</strong> {report.patient.name || "-"}</p>
            <p><strong>나이:</strong> {report.patient.age ?? "-"}</p>
            <p><strong>성별:</strong> {report.patient.gender === "M" ? "남성" : report.patient.gender === "F" ? "여성" : "-"}</p>
            <p><strong>암 종류:</strong> {report.patient.cancerType || "-"}</p>
            <p><strong>병기:</strong> {report.patient.stage || "-"}</p>
          </div>
        </div>
      )}

      {/* 치료 플랜 제안 */}
      {report && Array.isArray(report.plans) && (
        <div style={{ marginTop: 20 }}>
          <h2>치료 플랜 제안</h2>
          <ul>
            {report.plans.map((plan, idx) => (
              <li key={idx}>
                💊 {plan.name} — 확률: {Number(plan.probability).toFixed(1)}%{" "}
                {plan.recommended ? "(추천)" : ""}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 버튼 영역 */}
      <div style={{ marginTop: 20, display: "flex", gap: "10px", flexWrap: "wrap" }}>
        {onBackToSimulation && <button onClick={onBackToSimulation}>이전</button>}
        <button onClick={handleSaveServer}>결과 서버에 저장</button>
        <button onClick={handlePrint}>결과 출력</button>
        <button onClick={onRestart}>다시 시작</button>
      </div>
    </div>
  );
}
