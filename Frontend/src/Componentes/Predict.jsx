import { useState, useEffect } from "react";
import { predictFakeNews, getModelMetrics, getTopWords, checkInfluentialWords } from "../api";
import "./Predict.css";

const Predict = () => {
  const [noticias, setNoticias] = useState([{ Titulo: "", Descripcion: "" }]);
  const [resultados, setResultados] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [topWords, setTopWords] = useState([]);
  const [alertWords, setAlertWords] = useState([]);

  useEffect(() => {
    const fetchMetricsAndWords = async () => {
      const metricsData = await getModelMetrics();
      setMetrics(metricsData);

      const wordsData = await getTopWords();
      setTopWords(wordsData.words);
    };

    fetchMetricsAndWords();
  }, []);

  const handleChange = (index, field, value) => {
    const updatedNoticias = [...noticias];
    updatedNoticias[index][field] = value;
    setNoticias(updatedNoticias);
  };

  const handleAddNoticia = () => {
    setNoticias([...noticias, { Titulo: "", Descripcion: "" }]);
  };

  const handleRemoveNoticia = (index) => {
    setNoticias(noticias.filter((_, i) => i !== index));
  };


  const handlePredict = async () => {
    // Hacer la predicción
    const predicciones = await predictFakeNews(noticias);
    setResultados(predicciones);
  
    const wordsFoundPromises = noticias.map((noticia) => checkInfluentialWords(noticia));
    const wordsFoundResponses = await Promise.all(wordsFoundPromises);
  
    console.log("Palabras influyentes encontradas por cada noticia:", wordsFoundResponses);
  
    const matchedWords = wordsFoundResponses.flat(); 
  
    setAlertWords(matchedWords); 
  };
  
  

  const getRecommendation = (prediction, probability) => {
    if (prediction == 1) {
      return {
        mensaje: "✅ La noticia parece ser real.",
        recomendacion: [
          "Para periodistas: Aún así, sugerimos verificar con fuentes oficiales para mayor seguridad.",
          "Para agencias gubernamentales: Considera emitir un comunicado oficial para reforzar la información.",
          "Para el público: La fuente puede ser confiable, pero siempre es bueno contrastar con otras fuentes."
        ],
        color: "#4caf50"
      };
    } else {
      return {
        mensaje: "❌ La noticia podría ser falsa.",
        recomendacion: [
          "Para periodistas: Investiga fuentes primarias y contacta expertos antes de difundir.",
          "Para agencias gubernamentales: Considera emitir una alerta oficial y aclarar la información.",
          "Para el público: Verifica la información en fuentes confiables antes de compartirla."
        ],
        color: "#e74c3c"
      };
    }
  };

  return (
    <div className="predict-container">
      <div className="predict-box">
        <h2>🔍 Detección de Fake News</h2>

        {metrics && (
  <p className="metrics-info">
    <strong>Métricas del modelo:</strong><br />
    * <strong>Precisión:</strong> {metrics.precision.toFixed(4)}   
      (Indica qué tan bien el modelo evita falsos positivos, es decir, cuántas de las noticias clasificadas como falsas realmente lo son).<br /><br></br>
    * <strong>Recall:</strong> {metrics.recall.toFixed(4)}  
     (Mide la capacidad del modelo para detectar todas las Fake News, evitando falsos negativos).<br /><br></br>
    * <strong>F1-score:</strong> {metrics.f1_score.toFixed(4)}  
     (Es un equilibrio entre precisión y recall; lo ideal es que no haya mucha diferencia entre ellos para evitar sobreajuste).<br /><br /><br></br>
    - Un modelo equilibrado debe tener valores de precisión y recall similares para evitar sesgos y mejorar la confiabilidad en la detección de Fake News.
  </p>
)}


        {topWords.length > 0 && (
          <p className="top-words-info">
            -  <strong>Top 10 palabras influyentes en Fake News:</strong> {topWords.slice(0, 10).join(", ")}
          </p>
        )}

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
            {noticias.length > 1 && (
              <button className="remove-button" onClick={() => handleRemoveNoticia(index)}>Eliminar noticia</button>
            )}
          </div>
        ))}

        <button className="add-button" onClick={handleAddNoticia}>+ Agregar otra noticia</button>
        <button className="predict-button" onClick={handlePredict}>Predecir</button>

        <div className="results">
          <h2>Resultados:</h2>

          {alertWords.length > 0 && (
            <p className="alert-words">
             ⚠️ Se encontraron palabras clave dentro de las 1,500 más influyentes en Fake News:  <strong>{alertWords.join(", ")}</strong>
            </p>
          )}

          {resultados.length === 0 ? (
            <p className="no-results">Ingresa una noticia para analizar...</p>
          ) : (
            resultados.map((resultado, index) => {
              const { mensaje, recomendacion, color } = getRecommendation(resultado.Prediccion.prediction, resultado.Prediccion.probability);

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
