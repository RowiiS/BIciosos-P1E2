import { useState } from "react";

const Retrain = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Por favor selecciona un archivo.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setMessage("Modelo reentrenado con éxito.");
      } else {
        const errorData = await response.json();
        setMessage(` Error: ${errorData.detail || "No se pudo reentrenar."}`);
      }
    } catch (error) {
      setMessage("Error al conectar con el servidor.");
    }
  };

  return (
    <div>
      <h2>Reentrenar Modelo</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload}> Subir y Reentrenar</button>
      {message && <p>{message}</p>}
    </div>
  );
};

export default Retrain;
