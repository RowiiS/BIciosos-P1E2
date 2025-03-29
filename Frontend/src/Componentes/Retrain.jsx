import { useState, useEffect } from "react";
import "./Retrain.css";

const Retrain = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [updateTrigger, setUpdateTrigger] = useState(false);

  useEffect(() => {
    if (updateTrigger) {
      fetch("http://localhost:8000/model-metrics")
        .then((res) => res.json())
        .then((data) => {
          console.log("Métricas después de reentrenar:", data);
          setMetrics(data.metrics || data); // Ajuste por si la estructura cambia
        })
        .catch((err) => console.error("Error obteniendo métricas:", err));
      setUpdateTrigger(false);
    }
  }, [updateTrigger]);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setMessage("");
    setMetrics(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Por favor selecciona un archivo CSV para reentrenar el modelo.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMetrics(null);

    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setMessage("Modelo reentrenado con éxito. Esperando métricas...");
        setTimeout(() => setUpdateTrigger(true), 3000); // Esperar 3s antes de pedir métricas
      } else {
        const errorData = await response.json();
        setMessage(`Error: ${errorData.detail || "No se pudo reentrenar el modelo."}`);
      }
    } catch (error) {
      setMessage("Error al conectar con el servidor.");
    }

    setLoading(false);
  };

  return (
    <div className="retrain-container">
      <div className="retrain-box">
        <h2>Reentrenar Modelo</h2>
        <p className="retrain-description">
          Sube un archivo <strong>CSV</strong> con nuevos datos para mejorar la precisión del modelo de detección de Fake News.
        </p>

        <div className="csv-format">
          <h3>Formato necesario del CSV:</h3>
          <p><strong>Separador:</strong> Punto y coma <code>;</code></p>
          <p><strong>Columnas:</strong> <code>Label;Titulo;Descripcion</code></p>
          <p><strong>Ejemplo:</strong></p>
          <pre>
Label;Titulo;Descripcion<br></br>
1;The Guardian va con Sánchez;Europa necesita que su apuesta dé sus frutos<br></br>
0;Puigdemont: No sería ninguna tragedia;Ha desdramatizado un posible fracaso
          </pre>
          <p><strong>0:</strong> Noticia Falsa &nbsp;&nbsp;|&nbsp;&nbsp; <strong>1:</strong> Noticia Real</p>
        </div>

        <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />
        <button className="upload-button" onClick={handleUpload} disabled={loading}>
          {loading ? "Reentrenando..." : "Subir y Reentrenar"}
        </button>

        {message && <p className="message">{message}</p>}

        {metrics && metrics.precision && (
          <div className="metrics-box">
            <h3>Métricas del Modelo</h3>
            <p><strong>Precisión:</strong> {metrics.precision.toFixed(4)}</p>
            <p><strong>Recall:</strong> {metrics.recall.toFixed(4)}</p>
            <p><strong>F1-score:</strong> {metrics.f1_score.toFixed(4)}</p>

            <div className="metrics-info">
  <h4>¿Cómo interpretar estas métricas?</h4>
  <p>
    - <strong>Precisión (Precision):</strong> Mide cuántas de las noticias clasificadas como reales o falsas fueron correctamente identificadas.  
    Un valor alto significa que el modelo comete pocos <strong>falsos positivos</strong>, es decir, que casi no etiqueta noticias reales como falsas.  
    Por ejemplo, si una noticia real de un periódico reconocido es clasificada como Fake News, sería un falso positivo.  
  </p>
  <p>
    - <strong>Recall (Sensibilidad):</strong> Indica qué porcentaje de todas las Fake News existentes fueron correctamente detectadas.  
    Un recall bajo significa que hay <strong>falsos negativos</strong>, es decir, que el modelo deja pasar noticias falsas como si fueran reales.  
    Por ejemplo, si un artículo con información manipulada es clasificado como real, sería un falso negativo.  
  </p>
  <p>
    - <strong>F1-score:</strong> Es un equilibrio entre precisión y recall. Si la precisión es muy alta pero el recall es bajo,  
    significa que el modelo es demasiado estricto y deja pasar muchas Fake News. Si el recall es alto pero la precisión es baja,  
    el modelo detecta muchas Fake News, pero con demasiados errores, clasificando noticias reales como falsas.  
  </p>
  <p>
    Recuerda:  Lo ideal es que estas métricas sean lo más cercanas posible a <strong>1.0</strong>,  
    pero en la práctica es imposible alcanzar valores perfectos sin afectar el equilibrio.  
    Un buen modelo debe minimizar tanto los falsos positivos como los falsos negativos sin comprometer demasiado ninguna de las dos métricas.  
  </p>
</div>

          </div>
        )}
      </div>
    </div>
  );
};

export default Retrain;
