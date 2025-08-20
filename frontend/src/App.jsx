import { useState } from "react";
import Home from "./components/Home.jsx";
import PatientInput from "./components/PatientInput.jsx";
import VirtualPatient from "./components/VirtualPatient.jsx";
import Simulation from "./components/Simulation.jsx";
import Summary from "./components/Summary.jsx";

export default function App() {
  const [step, setStep] = useState(0);
  const [patientData, setPatientData] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [reportId, setReportId] = useState(null);

  const restart = () => {
    setStep(0);
    setPatientData(null);
    setSimulationResult(null);
    setSelectedScenario(null);
    setReportId(null);
  };

  return (
    <>
      {step === 0 && (
        <div className="start-screen">
          <h1 className="title">CanSim 시뮬레이터</h1>
          <button className="start-btn" onClick={() => setStep(1)}>
            시뮬레이션 시작
          </button>
        </div>
      )}

      {step === 1 && (
        <PatientInput
          onNext={() => setStep(2)}
          setPatientData={setPatientData}
          setSimulationResult={setSimulationResult}
          setReportId={setReportId}
        />
      )}

      {step === 2 && (
        <VirtualPatient
          patientData={patientData || {}}
          onPrev={() => setStep(1)}
          onNext={() => setStep(3)}
        />
      )}

      {step === 3 && (
        <Simulation
          patientData={patientData}
          simulationResult={simulationResult}
          onSelectScenario={(s) => {
            setSelectedScenario(s);
            setStep(4);
          }}
        />
      )}

      {step === 4 && (
        <Summary
          patientData={patientData}
          scenario={selectedScenario}
          reportId={reportId}
          onRestart={restart}
          onBackToSimulation={() => setStep(3)}
        />
      )}
    </>
  );
}