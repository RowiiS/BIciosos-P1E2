import { useState } from "react";
import { predictFakeNews } from "../api";

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

  return (
    <div style={{ maxWidth: "600px", margin: "auto", textAlign: "center" }}>
      <h2>Detección de Fake News</h2>

      {noticias.map((noticia, index) => (
        <div key={index} style={{ marginBottom: "10px" }}>
          <input
            type="text"
            placeholder="Título"
            value={noticia.Titulo}
            onChange={(e) => handleChange(index, "Titulo", e.target.value)}
          />
          <textarea
            placeholder="Descripción"
            value={noticia.Descripcion}
            onChange={(e) => handleChange(index, "Descripcion", e.target.value)}
          />
        </div>
      ))}

      <button onClick={handlePredict}>Predecir</button>

      <h2>Resultados:</h2>
      {resultados.map((resultado, index) => (
        <div key={index} style={{ border: "1px solid gray", padding: "10px", margin: "10px 0" }}>
          <p><strong>{resultado.Titulo}</strong></p>
          <p>{resultado.Descripcion}</p>
          <p>Predicción: {resultado.Prediccion.prediction === 1 ? "Real" : "Falsa"}</p>
          <p>Probabilidad: {(resultado.Prediccion.probability * 100).toFixed(2)}%</p>
        </div>
      ))}
    </div>
  );
};

export default Predict;
