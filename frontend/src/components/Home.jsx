export default function Home({ onNext }) {
  return (
    <div className="container">
      <h1>CanSim</h1>
      <p>환자 맞춤형 암 치료 시뮬레이터</p>
      <button onClick={onNext}>시뮬레이션 시작</button>
    </div>
  );
}
