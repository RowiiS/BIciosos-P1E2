import { Link } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "./Home.css";

function Home() {
  return (
    <div className="home-container d-flex flex-column align-items-center justify-content-center">
     
      <h1 className="title text-center">Detecta Fake News de manera rápida y sencilla</h1>
      <p className="subtitle text-center">
        Nuestra herramienta utiliza Machine Learning para verificar la autenticidad de las noticias y mejorar su precisión mediante reentrenamiento.
      </p>

      
      <div className="row g-4 w-75 justify-content-center">
        <div className="col-md-5">
          <Link to="/predict" className="card-option">
            <h2>🔍 Verificar Noticias</h2>
            <p>Ingresa un titular y descripción para analizar su veracidad.</p>
          </Link>
        </div>

        <div className="col-md-5">
          <Link to="/retrain" className="card-option">
            <h2>📟 Reentrenar Modelo</h2>
            <p>Sube un archivo CSV con datos para mejorar la precisión del modelo.</p>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
