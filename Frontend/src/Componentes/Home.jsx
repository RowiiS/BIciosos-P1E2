import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div>
      <h1>Fake News Detector</h1>
      <nav>
        <ul>
          <li><Link to="/predict">Predicción</Link></li>
          <li><Link to="/retrain">Reentrenamiento</Link></li>
        </ul>
      </nav>
    </div>
  );
};

export default Home;
