import { useState } from "react";
import { predictFakeNews } from "../api";
import "./Predict.css";

const Predict = () => {
  const [noticias, setNoticias] = useState([{ Titulo: "", Descripcion: "" }]);
  const [resultados, setResultados] = useState([]);

  const handleChange = (index, field, value) => {
    const updatedNoticias = [...noticias];
    updatedNoticias[index][field] = value;
    setNoticias(updatedNoticias);
  };

  const handlePredict = async () => {
    const predicciones = await predictFakeNews(noticias);
    setResultados(predicciones);
  };

  const getRecommendation = (prediction) => {
    if (prediction === 1) {
      return {
        mensaje: "✅ La noticia parece ser real.",
        recomendacion: [
          "Para periodistas: Aún así, sugerimos verificar con fuentes oficiales que te permitan tener seguridad a la hora de publicar",
          "Para agencias gubernamentales: Considera emitir un comunicado oficial para reforzar la información",
            "Para el público: Considera que la fuente de la noticia puede ser confiable, pero siempre es bueno contrastar con otras fuentes",
        ],
        color: "#4caf50"
      };
    } else {
      return {
        mensaje: "❌ La noticia podría ser falsa.",
        recomendacion: [
          "Para periodistas: Investiga fuentes primarias y contacta expertos antes de difundir",
          "Para agencias gubernamentales: Considera emitir una alerta oficial y aclarar la información",
            "Para el público: Verifica la información en fuentes confiables antes de compartirla con tus familiares, amigos o conocidos",
        ],
        color: "#e74c3c"
      };
    }
  };

  return (
    <div className="predict-container">
      <div className="predict-box">
        <h2>🔍 Detección de Fake News</h2>
        <p className="predict-description">
          Ingresa el título y la descripción de una noticia en los campos siguientes y presiona <strong>"Predecir"</strong>.
          Nuestra Modelo de aprendizaje automático analizará la información y te dirá si es real o falsa con una probabilidad de certeza.
        </p>

        {noticias.map((noticia, index) => (
          <div key={index} className="input-group">
            <input
              type="text"
              placeholder="Escribe el título aquí..."
              value={noticia.Titulo}
              onChange={(e) => handleChange(index, "Titulo", e.target.value)}
            />
            <textarea
              placeholder="Escribe la descripción aquí..."
              value={noticia.Descripcion}
              onChange={(e) => handleChange(index, "Descripcion", e.target.value)}
            />
          </div>
        ))}

        <button className="predict-button" onClick={handlePredict}>Predecir</button>

        <div className="results">
          <h2>Resultados:</h2>
          {resultados.length === 0 ? (
            <p className="no-results">🔎 Ingresa una noticia para analizar...</p>
          ) : (
            resultados.map((resultado, index) => {
              const { mensaje, recomendacion, color } = getRecommendation(resultado.Prediccion.prediction);

              return (
                <div key={index} className="result-card" style={{ borderLeft: `5px solid ${color}` }}>
                  <p><strong>{resultado.Titulo}</strong></p>
                  <p>{resultado.Descripcion}</p>
                  <p style={{ fontWeight: "bold", color }}>{mensaje}</p>
                  <ul className="recommendations">
                    {recomendacion.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                  <p>Probabilidad: {(resultado.Prediccion.probability * 100).toFixed(2)}%</p>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default Predict;
