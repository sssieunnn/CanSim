// src/components/PatientInput.jsx
import { useState } from "react";
const API_BASE = "http://127.0.0.1:8000";

export default function PatientInput({ onNext, setPatientData, setSimulationResult, setReportId }) {
  const [form, setForm] = useState({
    name: "", age: "", gender: "",
    cancerType: "", stage: "", chemo: "",
    temperature: "", bloodPressure: "", pulse: "", extra: "",
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleNext = async () => {
    try {
      setLoading(true);
      const payload = {
        ...form,
        age: form.age !== "" ? Number(form.age) : null,
        pulse: form.pulse !== "" ? Number(form.pulse) : null,
      };
      setPatientData(payload);

      const res = await fetch(`${API_BASE}/api/simulate_viz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data; try { data = text ? JSON.parse(text) : null; } catch { data = text; }

      const rid = res.headers.get("X-Report-Id");
      if (rid) setReportId(rid);
      else if (data?.report_id) setReportId(data.report_id);

      if (!res.ok) {
        const msg = data?.detail
          ? (Array.isArray(data.detail)
              ? data.detail.map(d => `${(d.loc||[]).join(".")}: ${d.msg}`).join("\n")
              : JSON.stringify(data.detail))
          : (typeof data === "string" ? data : `HTTP ${res.status}`);
        throw new Error(msg);
      }

      setSimulationResult(data);
      onNext();
    } catch (err) {
      console.error("API 호출 실패:", err);
      alert(`시뮬레이션 호출 실패: ${err.message || String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>환자 데이터 입력</h1>
      <div className="form">
        <input name="name" type="text" placeholder="이름" value={form.name} onChange={handleChange} /><br />
        <input name="age" type="number" placeholder="나이" value={form.age} onChange={handleChange} /><br />
        <select name="gender" value={form.gender} onChange={handleChange}>
          <option value="">성별 선택</option><option value="M">남성</option><option value="F">여성</option>
        </select><br />
        <select name="cancerType" value={form.cancerType} onChange={handleChange} required>
          <option value="">암 종류 선택</option>
          <option value="폐암">폐암</option><option value="대장암">대장암</option>
          <option value="위암">위암</option><option value="전립선암">전립선암</option>
          <option value="유방암">유방암</option><option value="뇌교종">뇌교종</option>
        </select><br />
        <select name="stage" value={form.stage} onChange={handleChange}>
          <option value="">병기 선택</option><option value="I">I</option><option value="II">II</option>
          <option value="III">III</option><option value="IV">IV</option>
        </select><br />
        <input name="chemo" type="text" placeholder="항암제/요법(선택)" value={form.chemo} onChange={handleChange} /><br />
        <input name="temperature" type="text" placeholder="체온 (예: 36.8)" value={form.temperature} onChange={handleChange} /><br />
        <input name="bloodPressure" type="text" placeholder="혈압 (예: 120/80)" value={form.bloodPressure} onChange={handleChange} /><br />
        <input name="pulse" type="number" placeholder="맥박 (예: 72)" value={form.pulse} onChange={handleChange} /><br />
        <input name="extra" type="text" placeholder="추가사항 (선택)" value={form.extra} onChange={handleChange} /><br />

        <button type="button" onClick={handleNext} disabled={loading}>
          {loading ? "요청 중..." : "다음"}
        </button>
      </div>
    </div>
  );
}