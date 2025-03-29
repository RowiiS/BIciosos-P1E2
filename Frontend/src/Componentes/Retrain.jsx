import { useState } from "react";
import "./Retrain.css";

const Retrain = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setMessage("");
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Por favor selecciona un archivo CSV para reentrenar el modelo.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setMessage("✅ Modelo reentrenado con éxito.");
      } else {
        const errorData = await response.json();
        setMessage(`❌ Error: ${errorData.detail || "No se pudo reentrenar el modelo."}`);
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
          Sube un archivo <strong>CSV</strong> con nuevos datos para mejorar la precisión del modelo de detección de Fake News. Si el archivo es muy grande, ten paciencia, el reentrenamiento puede tardar un poco.
        </p>

        <div className="csv-format">
          <h3>Formato necesario del CSV:</h3>
          <p><strong>Separador:</strong> Punto y coma <code>;</code></p>
          <p><strong>Columnas:</strong> <code>Label;Titulo;Descripcion</code></p>
          <p><strong>Ejemplo:</strong></p>
          <pre>
Label;Titulo;Descripcion <br></br>
1;The Guardian va con Sánchez;Europa necesita que su apuesta dé sus frutos <br></br>
0;Puigdemont: No sería ninguna tragedia;Ha desdramatizado un posible fracaso
          </pre>
          <p><strong>🔴 0:</strong> Noticia Falsa &nbsp;&nbsp;|&nbsp;&nbsp; <strong>🟢 1:</strong> Noticia Real</p>
        </div>

        <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />
        <button className="upload-button" onClick={handleUpload} disabled={loading}>
          {loading ? "Reentrenando..." : " Subir y Reentrenar"}
        </button>

        {message && <p className="message">{message}</p>}
      </div>
    </div>
  );
};

export default Retrain;
