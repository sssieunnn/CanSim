export default function VirtualPatient({ patientData, onPrev, onNext }) {
  const labelMap = {
    name: "이름",
    age: "나이",
    gender: "성별",
    cancerType: "암 종류",
    stage: "병기",
    chemo: "항암제 및 요법",
    temperature: "체온",
    bloodPressure: "혈압",
    pulse: "맥박",
    extra: "추가사항",
  };

  // 값 가공(성별 같은 코드값 표기 정리)
  const formatValue = (key, value) => {
    if (value === null || value === undefined || value === "") return "-";
    if (key === "gender") {
      if (value === "M") return "남성";
      if (value === "F") return "여성";
    }
    return String(value);
  };

  // 표시 순서(원하는 순서로 정렬)
  const order = [
    "name", "age", "gender", "cancerType", "stage",
    "chemo", "temperature", "bloodPressure", "pulse", "extra",
  ];
  return (
    <div className="container">
      <h1>입력 정보 확인</h1>

      <div className="form" style={{ textAlign: "left", margin: "0 auto" }}>
        {order.map((key) => (
          <div key={key} style={{ margin: "8px 0" }}>
            <strong>{labelMap[key]} :</strong> {formatValue(key, patientData[key])}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <button onClick={onPrev}>정보 수정</button>
        <button onClick={onNext}>시뮬레이션 시작</button>
      </div>
    </div>
  );
}
