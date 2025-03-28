export const predictFakeNews = async (noticias) => {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ noticias }),
    });
  
    const data = await response.json();
    return data.resultados;
  };
  